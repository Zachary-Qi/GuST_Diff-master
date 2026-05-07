import os
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import logging
# from .ranger21 import Ranger


def train(
    model,
    args,
    train_loader,
    valid_loader=None,
    foldername="",
    resume_ckpt_path=None,
):
    """
    带断点恢复的训练循环。

    - 如果 resume_ckpt_path 是一个有效 checkpoint：
        * 恢复 model / optimizer / scheduler / best_valid_loss / epoch
        * 从下一轮 epoch 继续训练
    - 否则从头训练。

    每个 epoch 结束后会刷新 checkpoint_latest.pth（可续训）。

    训练结束后会保存：
      - model.pth          : 最后一轮权重
      - best_model.pth     : 验证集上最优权重
      - checkpoint_latest.pth : 方便继续训练的完整状态
    """

    # -------------------
    # 1. optimizer & scheduler
    # -------------------
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    # optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=0.0001)

    lr_scheduler = None
    if getattr(args, "is_lr_decay", False):
        p1 = int(0.75 * args.epochs)
        p2 = int(0.9 * args.epochs)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )

    # -------------------
    # 2. logging & path prep
    # -------------------
    if foldername != "":
        os.makedirs(foldername, exist_ok=True)

        final_model_path = os.path.join(foldername, "model.pth")        # 最后一轮
        best_model_path  = os.path.join(foldername, "best_model.pth")   # 最优
        log_path         = os.path.join(foldername, "train_model.log")

        logging.basicConfig(
            filename=log_path,
            level=logging.DEBUG
        )
    else:
        final_model_path = "model.pth"
        best_model_path  = "best_model.pth"

    # -------------------
    # 3. 尝试恢复断点 (resume)
    # -------------------
    start_epoch = 0
    best_valid_loss = 1e10
    best_state_dict = None  # 当前最优模型快照（state_dict）

    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        # 说明：这是我们自己保存的训练中断点，里面不仅有模型权重，还有优化器/scheduler状态等。
        # 这些对象是本地可信的，所以我们明确允许完整反序列化（weights_only=False）。
        ckpt = torch.load(
            resume_ckpt_path,
            map_location=model.device if hasattr(model, "device") else "cpu",
            weights_only=False,  # 显式声明，避免未来 PyTorch 默认行为变更带来的 break
        )

        # 模型参数
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])

        # optimizer
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        # scheduler
        if lr_scheduler is not None and "scheduler_state" in ckpt:
            lr_scheduler.load_state_dict(ckpt["scheduler_state"])

        # epoch
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

        # best_valid_loss
        if "best_valid_loss" in ckpt:
            best_valid_loss = ckpt["best_valid_loss"]

        # 如果之前已经训练过一阵并生成 best_model.pth，就把它载入为当前 best_state_dict
        if foldername != "":
            maybe_best_model = os.path.join(foldername, "best_model.pth")
            if os.path.isfile(maybe_best_model):
                try:
                    prev_best_state = torch.load(
                        maybe_best_model,
                        map_location=model.device if hasattr(model, "device") else "cpu",
                        weights_only=True,  # 这里是纯 state_dict()
                    )
                    best_state_dict = prev_best_state
                    print(f"[resume] loaded previous best_model.pth from {maybe_best_model}")
                    logging.info(f"[resume] loaded previous best_model.pth from {maybe_best_model}")
                except Exception as e:
                    print(f"[WARN] failed to load previous best_model.pth: {e}")
                    logging.warning(f"[WARN] failed to load previous best_model.pth: {e}")

        msg = (
            f"[resume] loaded checkpoint '{resume_ckpt_path}' "
            f"(start_epoch={start_epoch}, best_valid_loss={best_valid_loss})"
        )
        logging.info(msg)

    # -------------------
    # 4. 训练循环
    # -------------------
    valid_epoch_interval = args.valid_epoch_interval

    for epoch_no in range(start_epoch, args.epochs):
        avg_loss = 0.0
        model.train()

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                # 这里假设 model(batch) 返回的是当前 batch 的 loss
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

            logging.info(
                f"avg_epoch_loss:{avg_loss / batch_no}, epoch:{epoch_no}"
            )

        # scheduler 每个 epoch 更新
        if lr_scheduler is not None:
            lr_scheduler.step()

        # -------------------
        # 5. 验证：训练过半后，每 valid_epoch_interval 评一次
        # -------------------
        do_validate = (
            valid_loader is not None
            and (epoch_no + 1) % valid_epoch_interval == 0
            and (epoch_no + 1) > args.epochs * 0.5
        )

        if do_validate:
            model.eval()
            avg_loss_valid = 0.0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()

                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

                logging.info(
                    f"valid_avg_epoch_loss:{avg_loss_valid / batch_no}, epoch:{epoch_no}"
                )

            # 如果验证集更好了 → 更新 best_valid_loss / best_state_dict / tmp_model
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                logging.info(
                    f"best loss is updated to {avg_loss_valid / batch_no} at {epoch_no}"
                )

                # 历史留档：tmp_model{epoch}.pth
                if foldername != "":
                    tmp_path = os.path.join(foldername, f"tmp_model{epoch_no}.pth")
                    torch.save(model.state_dict(), tmp_path)

                # 更新当前最优权重到内存
                best_state_dict = model.state_dict()

        # -------------------
        # 6. 每个 epoch 的持续 checkpoint (可续训)
        # -------------------
        if foldername != "":
            checkpoint_latest = {
                "epoch": epoch_no,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss,
            }
            if lr_scheduler is not None:
                checkpoint_latest["scheduler_state"] = lr_scheduler.state_dict()

            torch.save(
                checkpoint_latest,
                os.path.join(foldername, "checkpoint_latest.pth"),
            )

    # -------------------
    # 7. 训练结束后保存最终版本
    # -------------------
    if foldername != "":
        # 最后一轮权重（不一定泛化最好）
        torch.save(model.state_dict(), final_model_path)

        # 验证集上最好的权重
        if best_state_dict is not None:
            torch.save(best_state_dict, best_model_path)
        else:
            # 没跑验证 / 没触发do_validate的 fallback：把最后一轮当 best
            torch.save(model.state_dict(), best_model_path)

# -------------------------------
# 低层工具函数（掩码 & 反标准化）
# -------------------------------
def unscale(x, scaler, mean):
    # x: tensor, 支持 broadcast
    return x * scaler + mean

def masked_sum(x, mask):
    return torch.sum(x * mask)

def masked_count(mask):
    return torch.sum(mask)

def safe_div(numer, denom, eps=1e-12):
    return numer / (denom + eps)

# -------------------------------
# 分位损失 / CRPS（矢量化高效版）
# -------------------------------
def quantile_loss(target, forecast_q, q: float, eval_points):
    """
    target, forecast_q, eval_points shape: (B, L, K)
    """
    # pinball loss（也叫 quantile loss）
    # loss = 2 * |(y_hat - y)| * ( I[y <= y_hat] - q )
    indicator = (target <= forecast_q).float()
    loss = torch.abs(forecast_q - target) * (indicator - q)
    # 注意上式在 y>y_hat 处为负值；与题主原实现一致：再取绝对值前面带 2。
    return 2.0 * masked_sum(torch.abs(loss), eval_points)

def calc_denominator(target, eval_points, eps=1e-12):
    return masked_sum(torch.abs(target), eval_points) + eps

def calc_quantile_CRPS(target, samples, eval_points, mean_scaler, scaler,
                       quantiles=None):
    """
    target:            (B, L, K) 已在 0-1 标准化
    samples:           (B, nsample, L, K) 也是标准化
    eval_points:       (B, L, K)
    返回：标量 CRPS
    """
    if quantiles is None:
        # 与原实现一致：0.05, 0.10, ..., 0.95
        quantiles = np.arange(0.05, 1.0, 0.05)

    # 反标准化到原始物理量
    target_u   = unscale(target, scaler, mean_scaler)        # (B, L, K)
    samples_u  = unscale(samples, scaler, mean_scaler)       # (B, ns, L, K)
    denom = calc_denominator(target_u, eval_points)

    CRPS = 0.0
    # 按分位数在 nsample 维上直接求分位（dim=1）
    for q in quantiles:
        q_pred = torch.quantile(samples_u, q, dim=1)         # (B, L, K)
        q_loss = quantile_loss(target_u, q_pred, q, eval_points)
        CRPS += q_loss / denom

    return (CRPS / len(quantiles)).item()

# -------------------------------
# 其他常用指标（均在反标准化后）
# -------------------------------
def mse_metric(pred, target, eval_points):
    sq = (pred - target) ** 2
    return safe_div(masked_sum(sq, eval_points), masked_count(eval_points)).item()

def rmse_metric(pred, target, eval_points):
    return np.sqrt(mse_metric(pred, target, eval_points))

def mae_metric(pred, target, eval_points):
    ae = torch.abs(pred - target)
    return safe_div(masked_sum(ae, eval_points), masked_count(eval_points)).item()

def mape_metric(pred, target, eval_points, eps=1e-6):
    # 仅统计 |target| > eps 的位置
    valid = (torch.abs(target) > eps).float() * eval_points
    pe = torch.abs((pred - target) / (torch.abs(target) + eps))
    return safe_div(masked_sum(pe, valid), masked_count(valid)).item()

def smape_metric(pred, target, eval_points, eps=1e-6):
    den = (torch.abs(target) + torch.abs(pred)).clamp_min(eps)
    smape = 2.0 * torch.abs(pred - target) / den
    return safe_div(masked_sum(smape, eval_points), masked_count(eval_points)).item()

def r2_metric(pred, target, eval_points, eps=1e-12):
    # 只对 eval_points 的位置计算
    mask = eval_points
    y = target * mask
    yhat = pred * mask

    # 加权均值
    cnt = masked_count(mask)
    y_mean = safe_div(torch.sum(y), cnt)

    ss_res = torch.sum(((yhat - y) ** 2))
    ss_tot = torch.sum(((y - y_mean) ** 2))
    return (1.0 - safe_div(ss_res, ss_tot + eps)).item()

# -------------------------------
# 指标累加器（封装）
# -------------------------------
class MetricsAccumulator:
    def __init__(self, scaler=1.0, mean_scaler=0.0, quantiles=None):
        self.scaler = scaler
        self.mean_scaler = mean_scaler
        self.quantiles = quantiles or np.arange(0.05, 1.0, 0.05)

        # 累加器
        self.num_points = 0.0
        self._mse_sum = 0.0
        self._mae_sum = 0.0
        self._mape_sum = 0.0
        self._mape_cnt = 0.0
        self._smape_sum = 0.0
        self._r2_ss_res = 0.0
        self._r2_ss_tot = 0.0

        # 为了最终一次性算 CRPS，需要把所有 target / eval_points / samples 留存
        self._targets = []
        self._eval_points = []
        self._samples = []  # (B, ns, L, K) 保存每个 batch 的采样

    @torch.no_grad()
    def update(self, samples, target, eval_points):
        """
        samples: (B, nsample, L, K) 标准化
        target:  (B, L, K)         标准化
        eval_points: (B, L, K)
        """
        # 反标准化
        target_u  = unscale(target, self.scaler, self.mean_scaler)
        median_u  = torch.median(unscale(samples, self.scaler, self.mean_scaler), dim=1).values  # (B, L, K)

        # 统计量
        cnt = masked_count(eval_points).item()
        self.num_points += cnt

        # MSE/MAE
        self._mse_sum += masked_sum((median_u - target_u) ** 2, eval_points).item()
        self._mae_sum += masked_sum(torch.abs(median_u - target_u), eval_points).item()

        # MAPE（仅统计 |y|>eps）
        eps = 1e-6
        valid = (torch.abs(target_u) > eps).float() * eval_points
        self._mape_sum += masked_sum(torch.abs((median_u - target_u) / (torch.abs(target_u) + eps)), valid).item()
        self._mape_cnt += masked_count(valid).item()

        # sMAPE
        den = (torch.abs(target_u) + torch.abs(median_u)).clamp_min(eps)
        self._smape_sum += masked_sum(2.0 * torch.abs(median_u - target_u) / den, eval_points).item()

        # R²（按掩码统计）
        y = target_u * eval_points
        yhat = median_u * eval_points
        y_mean = safe_div(torch.sum(y), masked_count(eval_points))
        self._r2_ss_res += torch.sum((yhat - y) ** 2).item()
        self._r2_ss_tot += torch.sum((y - y_mean) ** 2).item()

        # 为 CRPS 保存
        self._targets.append(target)
        self._eval_points.append(eval_points)
        self._samples.append(samples)

    @torch.no_grad()
    def compute(self):
        # 聚合张量用于 CRPS
        target_all = torch.cat(self._targets, dim=0) if self._targets else None
        eval_all   = torch.cat(self._eval_points, dim=0) if self._eval_points else None
        samples_all= torch.cat(self._samples, dim=0) if self._samples else None

        if target_all is not None:
            crps = calc_quantile_CRPS(
                target_all, samples_all, eval_all,
                mean_scaler=self.mean_scaler, scaler=self.scaler,
                quantiles=self.quantiles
            )
        else:
            crps = float('nan')

        mse  = self._mse_sum / max(self.num_points, 1.0)
        rmse = np.sqrt(mse)
        mae  = self._mae_sum / max(self.num_points, 1.0)
        mape = self._mape_sum / max(self._mape_cnt, 1.0)
        smape= self._smape_sum / max(self.num_points, 1.0)
        r2   = 1.0 - safe_div(torch.tensor(self._r2_ss_res), torch.tensor(self._r2_ss_tot) + 1e-12).item()

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "sMAPE": smape,
            "R2": r2,
            "CRPS": crps,
        }

# -------------------------------
# 评估主函数（调用封装后的指标器）
# -------------------------------
@torch.no_grad()
def evaluate(model, test_loader, nsample=100, scaler=1.0, mean_scaler=0.0,
             foldername=""):
    """
    推理/评估阶段：
    - 不做梯度
    - 统一反标准化后在 eval_points 上计算 MSE / RMSE / MAE / MAPE / sMAPE / R² / CRPS
    - 支持保存采样与指标
    """
    model.eval()

    all_target = []
    all_evalpoint = []
    all_observed_point = []
    all_generated_samples = []

    metrics_acc = MetricsAccumulator(scaler=scaler, mean_scaler=mean_scaler)

    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            samples, c_target, eval_points, observed_points = model.inference_step(test_batch, nsample)
            # 期望 shapes 对齐：
            # samples: (B, ns, L, K)；c_target/ eval_points/ observed_points: (B, L, K)
            samples = samples.permute(0, 1, 3, 2)       # (B, ns, L, K)
            c_target = c_target.permute(0, 2, 1)        # (B, L, K)
            eval_points = eval_points.permute(0, 2, 1)  # (B, L, K)
            observed_points = observed_points.permute(0, 2, 1)

            # 更新指标器（内部完成反标准化与统计）
            metrics_acc.update(samples, c_target, eval_points)

            # 日志展示使用当前累计结果
            current = metrics_acc.compute()
            it.set_postfix(
                ordered_dict={
                    "rmse_total": current["RMSE"],
                    "mae_total": current["MAE"],
                    "batch_no": batch_no,
                },
                refresh=True,
            )
            logging.info(f"rmse_total={current['RMSE']}")
            logging.info(f"mae_total={current['MAE']}")
            logging.info(f"batch_no={batch_no}")

            # 若需要保存中间结果
            all_target.append(c_target)
            all_evalpoint.append(eval_points)
            all_observed_point.append(observed_points)
            all_generated_samples.append(samples)

    # 拼接保存采样等中间结果
    all_target = torch.cat(all_target, dim=0) if all_target else None
    all_evalpoint = torch.cat(all_evalpoint, dim=0) if all_evalpoint else None
    all_observed_point = torch.cat(all_observed_point, dim=0) if all_observed_point else None
    all_generated_samples = torch.cat(all_generated_samples, dim=0) if all_generated_samples else None

    if foldername:
        with open(os.path.join(foldername, f"generated_outputs_nsample{nsample}.pk"), "wb") as f:
            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    all_evalpoint,
                    all_observed_point,
                    scaler,
                    mean_scaler,
                ],
                f,
            )

    # 计算最终指标
    results = metrics_acc.compute()

    if foldername:
        with open(os.path.join(foldername, f"result_nsample{nsample}.pk"), "wb") as f:
            pickle.dump(
                [
                    results["RMSE"],
                    results["MAE"],
                    results["CRPS"],
                    results["MSE"],
                    results["MAPE"],
                    results["sMAPE"],
                    results["R2"],
                ],
                f,
            )

    print("Metrics:")
    for k, v in results.items():
        print(f"{k}: {v}")
        logging.info(f"{k}={v}")

    return results


# def quantile_loss(target, forecast, q: float, eval_points) -> float:
#     return 2 * torch.sum(
#         torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
#     )


# def calc_denominator(target, eval_points):
#     return torch.sum(torch.abs(target * eval_points))


# def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
#     target = target * scaler + mean_scaler
#     forecast = forecast * scaler + mean_scaler

#     quantiles = np.arange(0.05, 1.0, 0.05)
#     denom = calc_denominator(target, eval_points)
#     CRPS = 0
#     for i in range(len(quantiles)):
#         q_pred = []
#         for j in range(len(forecast)):
#             q_pred.append(torch.quantile(forecast[j: j + 1], quantiles[i], dim=1))
#         q_pred = torch.cat(q_pred, 0)
#         q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
#         CRPS += q_loss / denom
#     return CRPS.item() / len(quantiles)


# def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
#     """
#     推理/评估阶段:
#     - 不做梯度
#     - 计算 RMSE / MAE / CRPS
#     - 保存结果到 foldername
#     """
#     with torch.no_grad():
#         model.eval()
#         mse_total = 0.0
#         mae_total = 0.0
#         evalpoints_total = 0.0

#         all_target = []
#         all_observed_point = []
#         all_observed_time = []
#         all_evalpoint = []
#         all_generated_samples = []

#         with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
#             for batch_no, test_batch in enumerate(it, start=1):
#                 output = model.inference_step(test_batch, nsample)

#                 (
#                     samples,
#                     c_target,
#                     eval_points,
#                     observed_points,
#                     # observed_time,
#                 ) = output

#                 # 对齐到指标计算期望的 shape
#                 samples = samples.permute(0, 1, 3, 2)       # (B, nsample, L, K)
#                 c_target = c_target.permute(0, 2, 1)        # (B, L, K)
#                 eval_points = eval_points.permute(0, 2, 1)  # (B, L, K)
#                 observed_points = observed_points.permute(0, 2, 1)

#                 samples_median = samples.median(dim=1)  # -> .values shape (B, L, K)

#                 all_target.append(c_target)
#                 all_evalpoint.append(eval_points)
#                 all_observed_point.append(observed_points)
#                 # all_observed_time.append(observed_time)
#                 all_generated_samples.append(samples)

#                 mse_current = (
#                     ((samples_median.values - c_target) * eval_points) ** 2
#                 ) * (scaler ** 2)
#                 mae_current = (
#                     torch.abs((samples_median.values - c_target) * eval_points)
#                 ) * scaler

#                 mse_total += mse_current.sum().item()
#                 mae_total += mae_current.sum().item()
#                 evalpoints_total += eval_points.sum().item()

#                 it.set_postfix(
#                     ordered_dict={
#                         "rmse_total": np.sqrt(mse_total / evalpoints_total),
#                         "mae_total": mae_total / evalpoints_total,
#                         "batch_no": batch_no,
#                     },
#                     refresh=True,
#                 )
#                 logging.info(
#                     "rmse_total={}".format(np.sqrt(mse_total / evalpoints_total))
#                 )
#                 logging.info(
#                     "mae_total={}".format(mae_total / evalpoints_total)
#                 )
#                 logging.info("batch_no={}".format(batch_no))

#         # 拼接
#         all_target = torch.cat(all_target, dim=0)
#         all_evalpoint = torch.cat(all_evalpoint, dim=0)
#         all_observed_point = torch.cat(all_observed_point, dim=0)
#         # all_observed_time = torch.cat(all_observed_time, dim=0)
#         all_generated_samples = torch.cat(all_generated_samples, dim=0)

#         # 保存中间采样结果
#         if foldername != "":
#             with open(
#                 os.path.join(foldername, f"generated_outputs_nsample{nsample}.pk"),
#                 "wb",
#             ) as f:
#                 pickle.dump(
#                     [
#                         all_generated_samples,
#                         all_target,
#                         all_evalpoint,
#                         all_observed_point,
#                         # all_observed_time,
#                         scaler,
#                         mean_scaler,
#                     ],
#                     f,
#                 )

#         # 计算 CRPS
#         CRPS = calc_quantile_CRPS(
#             all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
#         )

#         RMSE_val = np.sqrt(mse_total / evalpoints_total)
#         MAE_val = mae_total / evalpoints_total

#         # 结果输出
#         if foldername != "":
#             with open(
#                 os.path.join(foldername, f"result_nsample{nsample}.pk"),
#                 "wb",
#             ) as f:
#                 pickle.dump(
#                     [
#                         RMSE_val,
#                         MAE_val,
#                         CRPS,
#                     ],
#                     f,
#                 )

#         print("RMSE:", RMSE_val)
#         print("MAE:", MAE_val)
#         print("CRPS:", CRPS)
#         logging.info("RMSE={}".format(RMSE_val))
#         logging.info("MAE={}".format(MAE_val))
#         logging.info("CRPS={}".format(CRPS))
