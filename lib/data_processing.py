import os
import sys
import torch
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader, Dataset

# 获取当前文件所在目录的上一级路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.MetrLA_data import get_metrLa_data
from utils.PemsBay_data import get_pemsbay_data
from utils.PeMS0X_data import get_pems0X_data
from utils.imputation_pipeline import apply_missing_mask
# from data_utils.imputation_pipeline import apply_missing_mask_3d_all_outputs_3d


# ---------------------------------------------------------------------
# 可见性掩码（cond_mask）— 随机点状：只在 observed_mask==1 的位置抽样
# ---------------------------------------------------------------------
@torch.no_grad()
def get_randmask(observed_mask: torch.Tensor,
                 min_miss_ratio: float = 0.0,
                 max_miss_ratio: float = 1.0) -> torch.Tensor:
    """
    随机可见性掩码（cond_mask）：从 observed_mask==1 的位置里随机隐藏一定比例。
    返回 float32 张量，1=可见，0=隐藏。
    """
    if not (0.0 <= min_miss_ratio <= 1.0 and 0.0 <= max_miss_ratio <= 1.0 and max_miss_ratio >= min_miss_ratio):
        raise ValueError("min_miss_ratio/max_miss_ratio 必须在 [0,1] 且 min<=max")

    # 随机确定一次这批样本的缺失比例
    sample_ratio = float(np.random.rand()) * (max_miss_ratio - min_miss_ratio) + min_miss_ratio

    # 仅在原本可观测的位置上打乱
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    flat = rand_for_mask.reshape(-1)
    num_observed = int(observed_mask.sum().item())
    num_masked = int(round(num_observed * sample_ratio))
    if num_masked > 0:
        topk_idx = flat.topk(num_masked).indices
        flat[topk_idx] = -1  # 标记为“隐藏”
    cond_mask = (flat > 0).reshape(observed_mask.shape).float()
    return cond_mask


# ---------------------------------------------------------------------
# 可见性掩码（cond_mask）— 结构化块：纯 Torch 实现，支持 CPU/GPU
# ---------------------------------------------------------------------
# 其中的min_seq max_seq用来生成 cond_mask（模型前向“可见性”）。控制的是训练时遮掉给模型看的那段连续长度，用于自监督重建。
@torch.no_grad()
def get_block_mask(observed_mask: torch.Tensor,
                   target_strategy: str = 'block', min_seq: int = 12, max_seq: int = 24) -> torch.Tensor:
    """
    结构化可见性掩码（cond_mask）：在 observed_mask==1 的位置上成块隐藏。
    返回 float32 张量，1=可见，0=隐藏。

    当前实现：
      - 先用稀疏“起点”采样（每列约 0~15%），再扩张成固定长度的时间块（[12,24)）。
      - 叠加少量点状噪声（5%）增强随机性。
      - 若 target_strategy == 'hybrid'，以 30% 概率退化为随机点状（与 get_randmask 相同）。

    Tips:
      可扩展为更细策略：'block_time'/'block_sensor'/'block_mixed' 等。
    """
    if observed_mask.dim() != 2:
        raise ValueError("get_block_mask 期望 2D [T,S] 的 observed_mask。")

    T, S = observed_mask.shape

    # 1) 稀疏起点（点状），随后扩张为时间块
    sample_ratio = float(np.random.rand() * 0.15)  # 列内起点稀疏度
    mask = (torch.rand_like(observed_mask) < sample_ratio)  # bool [T,S]

    # min_seq, max_seq = 12, 24
    if max_seq < min_seq:
        max_seq = min_seq

    # 2) 扩张为连续时间块（每列内）
    for col in range(S):
        idxs = torch.nonzero(mask[:, col], as_tuple=False).flatten()  # 起点
        if idxs.numel() == 0:
            continue
        # 当前列统一块长（也可以改成每个起点不同）
        fault_len = min_seq + (int(np.random.randint(max(1, max_seq - min_seq))) if max_seq > min_seq else 0)
        for i in idxs.tolist():
            start = int(i)
            end = min(start + fault_len, T)
            mask[start:end, col] = True

    # 3) 叠加 5% 点状噪声隐藏
    rand_base_mask = (torch.rand_like(observed_mask) < 0.05)  # bool
    reverse_mask = mask | rand_base_mask                      # bool

    # 4) 可见=1/隐藏=0；只在原本可观测处生效
    block_mask = 1.0 - reverse_mask.to(torch.float32)         # [T,S]
    cond_mask = observed_mask.clone().to(torch.float32)

    # 5) hybrid：部分批次退化为随机点状
    if target_strategy == "hybrid" and np.random.rand() > 0.7:
        cond_mask = get_randmask(observed_mask, 0.0, 1.0)
    else:
        cond_mask = block_mask * cond_mask

    return cond_mask


# ---------------------------------------------------------------------
# 测试阶段：固定比例点状遮挡
# ---------------------------------------------------------------------
@torch.no_grad()
def get_test_randmask(observed_mask: torch.Tensor, missing_ratio: float) -> torch.Tensor:
    """
    基于 observed_mask 随机隐藏固定比例 missing_ratio（0~1）的点；返回 cond_mask（1=可见，0=隐藏）。
    """
    if not (0.0 <= missing_ratio <= 1.0):
        raise ValueError("missing_ratio 必须在 [0,1]")
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    flat = rand_for_mask.reshape(-1)
    num_observed = int(observed_mask.sum().item())
    num_masked = int(round(num_observed * missing_ratio))
    if num_masked > 0:
        flat[flat.topk(num_masked).indices] = -1
    cond_mask = (flat > 0).reshape(observed_mask.shape).float()
    return cond_mask


# ---------------------------------------------------------------------
# 滑窗数据集（区分 cond_mask 与 gt_mask 的作用）
# ---------------------------------------------------------------------
class WindowedSTDataset(Dataset):
    """
    将 (T, S) 的时空矩阵切成 [eval_length, S] 的滑动窗口，产出以下键：
      - observed_data: [T, S]  标准化后的观测；原始“无观测”位置已清零
      - observed_mask: [T, S]  原始观测掩码（1=有原始观测，0=原始无观测）
      - gt_mask:       [T, S]  “可监督真值”掩码（1=既非人为缺失，又非原始无观测）
      - cond_mask:     [T, S]  “可见性”掩码（1=前向时允许模型看见，0=对模型隐藏）
      - timepoints, cut_length, coeffs: 与原实现一致

    【重要概念区分】
      - cond_mask（可见性）：控制“模型在前向阶段能不能用到该点的信息”，用于构造自监督任务的可见子集。
      - gt_mask（可监督性）：控制“该点是否参与损失/评估”，仅在有可靠真值的点上统计指标，避免无真值位置污染评估。

    【为什么要区分？】
      - 如果把可见点也纳入监督，模型容易退化为“拷贝已见值”，学不到推断能力；
      - 如果把无真值的位置纳入监督/评估，指标会失真。
    """
    def __init__(
        self,
        observed_data: np.ndarray,
        observed_mask: np.ndarray,
        gt_mask: np.ndarray,
        eval_length: int = 24,
        mode: str = "train",
        target_strategy: str = "random",   # 'random' | 'block' | 'hybrid' 等
        missing_pattern: str = "block",    # 仅 test 可用：'point' 时可通过 missing_ratio 做点状遮挡
        missing_ratio: Optional[float] = None,
        is_interpolate: bool = False,      # True: coeffs 给出基于前值的简单插补
        min_seq: int = 24,
        max_seq: int = 24,
    ):
        assert mode in {"train", "valid", "test"}
        self.mode = mode
        self.eval_length = int(eval_length)
        if self.eval_length < 1:
            raise ValueError(f"eval_length 必须 >= 1，当前为 {self.eval_length}")

        self.target_strategy = target_strategy
        self.missing_pattern = missing_pattern
        self.missing_ratio = missing_ratio
        self.is_interpolate = is_interpolate
        
        self.min_seq = min_seq
        self.max_seq = max_seq

        # 缓存为 float32，便于后续 torch 计算
        self.observed_data = observed_data.astype(np.float32, copy=False)
        self.observed_mask = observed_mask.astype(np.float32, copy=False)
        self.gt_mask = gt_mask.astype(np.float32, copy=False)

        T = self.observed_data.shape[0]
        if self.eval_length > T:
            raise ValueError(f"eval_length({self.eval_length}) 不能大于序列长度 T({T})")

        cur_len = T - self.eval_length + 1

        # 测试集按整段切分（不重叠）；末尾不足一整段则补最后一窗
        if mode == "test":
            n_sample = T // self.eval_length
            use_index = list(range(0, self.eval_length * n_sample, self.eval_length))
            cut_length = [0] * len(use_index)
            if T % self.eval_length != 0:
                use_index.append(cur_len - 1)
                cut_length.append(self.eval_length - (T % self.eval_length))
        else:
            use_index = list(range(cur_len))
            cut_length = [0] * cur_len

        self.use_index = use_index
        self.cut_length = cut_length

    def __len__(self) -> int:
        return len(self.use_index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.use_index[i]
        sl = slice(idx, idx + self.eval_length)

        ob_data = self.observed_data[sl]  # [T, S] 标准化后的观测（原始无观测处=0）
        ob_mask = self.observed_mask[sl]  # [T, S] 原始观测掩码：1=原始有观测
        gt_mask = self.gt_mask[sl]        # [T, S] 真值监督掩码：1=可用于监督/评估

        ob_mask_t = torch.tensor(ob_mask, dtype=torch.float32)

        # ========== cond_mask 生成规则 ==========
        if self.mode == "train":
            # 训练阶段：人为“遮住”一部分原本可见的点，作为训练目标去重建
            if self.target_strategy != "random":
                cond_mask = get_block_mask(ob_mask_t, target_strategy=self.target_strategy, min_seq=self.min_seq, max_seq=self.max_seq)
            else:
                cond_mask = get_randmask(ob_mask_t)
        else:
            # 验证/测试：两种协议
            if self.mode == "test" and self.missing_ratio is not None and self.missing_pattern == "point":
                # 测试点状基准：固定比例 missing_ratio 的随机隐藏（仅在 ob_mask==1 中采样）
                cond_mask = get_test_randmask(observed_mask=ob_mask_t, missing_ratio=self.missing_ratio)
            else:
                # 公平评估：使用 gt_mask 作为可见性，仅在有真值处评估
                cond_mask = torch.tensor(gt_mask, dtype=torch.float32)

        sample = {
            "observed_data": ob_data.astype(np.float32),
            "observed_mask": ob_mask.astype(np.float32),
            "gt_mask": gt_mask.astype(np.float32),
            # "timepoints": np.arange(self.eval_length, dtype=np.float32),
            "cut_length": self.cut_length[i],
            "cond_mask": cond_mask.numpy().astype(np.float32),
        }

        
        # ========== 可选：仅对第0通道做“前值插补”系数 ==========
        if self.is_interpolate:
            tmp = torch.tensor(ob_data, dtype=torch.float32)  # (T,S,D)
            # 只插补第0通道，时间通道保持不变
            for t in range(1, tmp.shape[0]):
                # cond_mask[t]: (S,)  -> (S,1) 便于广播到 (S,D)
                m_t = cond_mask[t]  # (S,)
                # 仅第0通道：用 squeeze(-1) 与索引方式避免上次的广播错误
                tmp_val  = tmp[t, :, 0]      # (S,)
                prev_val = tmp[t-1, :, 0]    # (S,)
                tmp[t, :, 0] = torch.where(m_t == 0, prev_val, tmp_val)
            sample["coeffs"] = tmp.numpy()    # (T,S,D)
        else:
            sample["coeffs"] = None

        return sample


def get_similarity_metrla(dist, thr=0.1, force_symmetric=False, sparse=False):
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

# ---------------------------------------------------------------------
# 数据读取 + 缺失采样 + 滑窗 + DataLoader
# ---------------------------------------------------------------------
def get_target_data(args):
    """
    返回：train_loader, valid_loader, test_loader, std_t, mean_t
    其中 std_t/mean_t shape=[S] 的 torch.float32 张量（在 CPU 上，便于后续 to(device)）
    """
    # 缺失形态对应的采样超参
    if args.missing_pattern == 'point':
        p_fault, p_noise = 0.0, 0.25
    elif args.missing_pattern == 'block':
        p_fault, p_noise = 0.0015, 0.05
    elif args.missing_pattern == 'sparse':
        p_fault, p_noise = 0.0, 0.9
    else:
        raise ValueError(f"Invalid missing pattern: {args.missing_pattern}.")

    # 数据集选择
    if args.dataset_name == 'MetrLA':
        args.seed = 9101112
        obj_data, mean, std = get_metrLa_data(args)
    elif args.dataset_name == 'PemsBay':
        args.seed = 9101112
        obj_data, mean, std = get_pemsbay_data(args)
    elif args.dataset_name in {'PeMS03', 'PeMS04', 'PeMS07', 'PeMS08'}:
        args.seed = 56789
        obj_data, mean, std = get_pems0X_data(args)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}.")

    # 缺失采样（在原始数据上造 eval_mask / gt_mask，并切分 train/val/test）
    # 其中的min_seq max_seq用来生成 eval_mask / gt_mask（评估或监督的“真值可用区域”）。控制的是评估/监督环节的人为连续缺失长度。
    masked_res = apply_missing_mask(
        args, obj_data, p_fault, p_noise, mean, std,
        min_seq=args.min_seq, max_seq=args.max_seq, seed=args.seed
    )

    train = masked_res["train"]
    val = masked_res["val"]
    test = masked_res["test"]

    # print("train——observed_data:", train["observed_data"].shape)  # -> (23990, 207, 3)
    # print("val——observed_data:", val["observed_data"].shape)  # -> (3427, 207, 3)
    # print("test——observed_data:", test["observed_data"].shape)  # -> (6855, 207, 3)


    # 滑窗数据集
    ds_train = WindowedSTDataset(
        train["observed_data"], train["observed_mask"], train["gt_mask"],
        eval_length=args.eval_length, mode="train",
        target_strategy=args.target_strategy, is_interpolate=args.is_interpolate, min_seq=args.min_seq, max_seq=args.max_seq
    )
    ds_valid = WindowedSTDataset(
        val["observed_data"], val["observed_mask"], val["gt_mask"],
        eval_length=args.eval_length, mode="valid",
        target_strategy=args.target_strategy, is_interpolate=args.is_interpolate, min_seq=args.min_seq, max_seq=args.max_seq
    )
    ds_test = WindowedSTDataset(
        test["observed_data"], test["observed_mask"], test["gt_mask"],
        eval_length=args.eval_length, mode="test",
        target_strategy=args.target_strategy, is_interpolate=args.is_interpolate,
        missing_pattern=args.missing_pattern, missing_ratio=None, min_seq=args.min_seq, max_seq=args.max_seq
    )

    # DataLoader（如需可复现，可加 generator / worker_init_fn）
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    valid_loader = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # scalers（优先采用显示传入；否则从 meta 里拿）
    meta = masked_res.get("meta", {})
    mean_vec = mean if mean is not None else meta.get("mean", None)
    std_vec  = std  if std  is not None else meta.get("std",  None)

    mean_t = torch.from_numpy(np.asarray(mean_vec).reshape(-1)).float() if mean_vec is not None else None
    std_t  = torch.from_numpy(np.asarray(std_vec ).reshape(-1)).float() if std_vec  is not None else None

    # obj_adj = get_similarity_metrla(obj_adj, thr=0.1, force_symmetric=True, sparse=False)
    
    return train_loader, valid_loader, test_loader, std_t, mean_t


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == '__main__':

    seed = 9101112

    parser = ArgumentParser()
    # 数据集/缺失采样
    parser.add_argument("--dataset_name", type=str, default="MetrLA")
    parser.add_argument("--missing_pattern", type=str, default="point", choices=["point", "block", "sparse"])
    parser.add_argument("--min_seq", type=int, default=12)
    parser.add_argument("--max_seq", type=int, default=12 * 4)
    parser.add_argument("--seed", type=int, default=seed)

    # 切分与滑窗
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--eval_length", type=int, default=24,
                        help="每个样本的时间窗口长度（步数），例如24表示2小时@5min间隔")

    # 训练相关
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_strategy", type=str, default="block",
                        choices=["random", "block", "hybrid"])

    # 是否使用前值插补（更稳的 argparse 写法是 action='store_true'）
    parser.add_argument("--is_interpolate", type=bool, default=True)

    args = parser.parse_args()

    # 运行一次自检
    tl, vl, te, std_t, mean_t = get_target_data(args)
    print("OK: loaders ready.",
          f"std={None if std_t is None else tuple(std_t.shape)}, "
          f"mean={None if mean_t is None else tuple(mean_t.shape)}")
    b = next(iter(tl))
    print("train batch keys:", b.keys())
    for k in ["observed_data", "observed_mask", "gt_mask", "cond_mask"]:
        x = b[k]
        print(f"{k:>14}: shape={np.array(x).shape if isinstance(x, np.ndarray) else (x.shape if hasattr(x,'shape') else type(x))}")

    # 1) 'observed_data' : torch.FloatTensor，形状 [B, T, S]
    #    - 标准化后的观测值（通常是 (x - mean) / std）。
    #    - 对于“原始无观测”的位置已置为 0，避免信息泄漏。
    #    - B: batch size；T: 窗口步数（eval_length）；S: 传感器/变量数。
    
    # 2) 'observed_mask' : torch.FloatTensor（0/1），形状 [B, T, S]
    #    - 原始观测掩码：1 表示该时刻该传感器在原始数据中“有观测”，0 表示原始就缺失。
    #    - 只反映数据本身，不包含训练时额外制造的遮挡。
    
    # 3) 'gt_mask' : torch.FloatTensor（0/1），形状 [B, T, S]
    #    - 可监督真值掩码（ground-truth mask）。
    #    - 1 表示该位置既有原始观测、又没有被“评估用遮挡”抹掉，可用作监督/评估真值；
    #      0 表示不可用于计算损失/指标。
    #    - 典型公式：gt_mask = 1 - (eval_mask | (1 - observed_mask))
    #
    # 4) 'cond_mask' : torch.FloatTensor（0/1），形状 [B, T, S]
    #    - 可见性掩码（conditioning mask），控制“前向时模型能看到哪些点”。
    #    - 训练阶段：由策略（random/block/hybrid 等）生成，在 observed_mask==1 的位置里再隐藏一部分，
    #      迫使模型用剩余可见点重建被遮住的点（自监督）。
    #    - 验证/测试（默认）：通常等于 gt_mask，只在有真值的位置可见并评估，保证公平；
    #      若设置了 missing_ratio 且 missing_pattern='point'，则会在可观测处额外随机隐藏固定比例形成基准。
    #
    # 5) 'timepoints' : torch.FloatTensor，形状 [T]（或 [B, T]）
    #    - 时间坐标（0,1,2,...,T-1），供显式时间建模（如 Neural CDE/ODE、时间编码）。
    #
    # 6) 'cut_length' : int（或 torch.Tensor 标量），长度为 B（或每样本一个标量）
    #    - 仅测试阶段末尾不足整窗时非零，记录该样本右端需要“补齐”的时长，评估时应忽略补齐段。
    #    - 训练/验证阶段通常为 0。
    #
    # 7) 'coeffs' : torch.FloatTensor 或 None
    #    - 当 is_interpolate=True 时提供，用“前值填充”的临时插补结果/系数（某些连续时间模型需要）；
    #      不作为监督目标。否则为 None。
    
    



    