import torch
import numpy as np
import torch.nn as nn

from .modules.guided_diffusion_unet import GuidedDiffusionUNet


class TimeFeatsEmbed(nn.Module):
    """
    构造时空条件特征 (time-of-day, day-of-week, sensor-id)
    输出形状: [B, C_out, S, T]
    """
    def __init__(self, steps_per_day=288, e_tod=16, e_dow=8,
                 num_sensors=207, e_feat=16, proj_out=64):
        super().__init__()
        self.steps_per_day = steps_per_day

        # 时间步 (一天内第几个5min)
        self.tod_emb = nn.Embedding(steps_per_day, e_tod)
        # 星期几
        self.dow_emb = nn.Embedding(7, e_dow)
        # 传感器ID嵌入
        self.feat_emb = nn.Embedding(num_sensors, e_feat)

        # 最后把 [tod||dow||feat] 投影到统一通道数 proj_out
        self.proj_out = nn.Conv2d(
            in_channels=e_tod + e_dow + e_feat,
            out_channels=proj_out,
            kernel_size=1
        )

    def forward(self, x):
        # 期望输入 x: [B, S, T, D]
        B, S, T, D = x.shape

        # x[..., 1] 假设是归一化的 "time-of-day", in [0,1)
        tod_idx = (x[..., 1] * self.steps_per_day).round().clamp(
            0, self.steps_per_day - 1
        ).long()  # [B, S, T]

        # x[..., 2] 假设是归一化的 "day-of-week", in [0,1), 映射到0..6
        dow_idx = (x[..., 2] * 7).round().clamp(0, 6).long()  # [B, S, T]

        # 查embedding
        tod_e = self.tod_emb(tod_idx)  # [B, S, T, e_tod]
        dow_e = self.dow_emb(dow_idx)  # [B, S, T, e_dow]

        # 每个sensor分配一个固定向量，然后广播到 [B, S, T, e_feat]
        feat_e = self.feat_emb(torch.arange(S, device=x.device))  # [S, e_feat]
        feat_e = feat_e.view(1, S, 1, -1).expand(B, S, T, -1)

        # 拼接 -> [B, S, T, e_tod+e_dow+e_feat]
        side = torch.cat([tod_e, dow_e, feat_e], dim=-1)

        # [B, C, S, T]
        side = side.permute(0, 3, 1, 2).contiguous()

        # 1x1 conv 投影到指定维度
        side = self.proj_out(side)  # [B, proj_out, S, T]

        return side


class GuSTDiffCore(nn.Module):
    """
    核心扩散插补模型，不直接依赖 dataloader 的 batch 字段命名。
    - 负责:
      * 构造扩散噪声输入给 GuidedDiffusionUNet
      * 计算训练/验证的loss
      * 运行反向扩散采样生成插补
      * 推理(inference_step)时打包输出
    """
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.target_dim = args.nodes          # 传感器数量 K
        self.seq_len = args.eval_length       # 序列长度 L

        self.emb_time_dim = args.timeemb
        self.emb_feature_dim = args.featureemb
        self.is_unconditional = args.is_unconditional
        self.target_strategy = args.target_strategy
        self.use_guide = args.use_guide

        self.cde_output_channels = args.channels
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        # 输入通道: 2 if (cond, noisy_target) or 1 if guided variant
        input_dim = 2
        self.diffmodel = GuidedDiffusionUNet(
            args, input_dim, self.target_dim,
            self.use_guide, self.emb_total_dim
        )

        # diffusion schedule
        self.num_steps = args.num_steps
        if args.schedule == "quad":
            beta = np.linspace(args.beta_start ** 0.5,
                               args.beta_end ** 0.5,
                               self.num_steps) ** 2
        elif args.schedule == "linear":
            beta = np.linspace(args.beta_start,
                               args.beta_end,
                               self.num_steps)
        else:
            raise ValueError(f"Unknown schedule {args.schedule}")
        self.beta = beta

        self.alpha_hat = 1 - self.beta              # \bar{alpha}_t (1 - beta_t)
        self.alpha = np.cumprod(self.alpha_hat)     # prod_t \bar{alpha}_t
        self.alpha_torch = torch.tensor(self.alpha).float().to(
            self.device
        ).unsqueeze(1).unsqueeze(1)                 # [num_steps,1,1]

        # 时空条件嵌入
        self.Temb = TimeFeatsEmbed(
            steps_per_day=288,
            e_tod=16,
            e_dow=8,
            num_sensors=self.target_dim,
            e_feat=16,
            proj_out=self.emb_time_dim + self.emb_feature_dim,
        )

    # ---------- loss 相关 ----------

    def loss_single_timestep(
        self,
        observed_data,   # [B,K,L] 归一化后的观测值
        cond_mask,       # [B,K,L] 训练时: 模型能看到的位置(条件); eval时: 指定
        observed_mask,   # [B,K,L] 原始可观测/有真值的位置
        side_info,       # [B,C,S,T] 时空条件嵌入
        itp_info,        # [B,1,K,L] 或 None，引导插值
        is_train,
        set_t=-1,
    ):
        """
        对单个扩散时间步 t 计算 MSE loss。
        训练时随机 t，验证时走固定 t。
        """
        B, K, L = observed_data.shape

        if is_train != 1:  # 验证/推理loss
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)

        # 加噪
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + \
                     (1.0 - current_alpha) ** 0.5 * noise

        # 构造diffusion模型输入
        total_input = self.build_diffusion_model_input(
            noisy_data, observed_data, cond_mask
        )

        # 如果不用guide，就用条件观测替代
        guided_itp = itp_info
        if not self.use_guide:
            guided_itp = cond_mask * observed_data  # [B,K,L]

        # 预测噪声
        predicted = self.diffmodel(total_input, side_info, t, guided_itp)

        # 只在被mask掉的目标位置上监督
        target_mask = observed_mask - cond_mask       # [B,K,L]
        residual = (noise - predicted) * target_mask  # [B,K,L]
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def loss_average_over_schedule(
        self,
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        itp_info,
        is_train,
    ):
        """
        验证时：遍历所有扩散步 t，取平均loss。
        """
        loss_sum = 0
        for t in range(self.num_steps):
            loss_t = self.loss_single_timestep(
                observed_data,
                cond_mask,
                observed_mask,
                side_info,
                itp_info,
                is_train,
                set_t=t,
            )
            # detach() 避免梯度积累；反正验证不反传
            loss_sum += loss_t.detach()
        return loss_sum / self.num_steps

    # ---------- 构造扩散网络输入 ----------

    def build_diffusion_model_input(self, noisy_data, observed_data, cond_mask):
        """
        根据 is_unconditional / use_guide 构造 diffmodel 的输入通道:
        - unconditional: just noisy_data
        - conditional (no_guide): concat[ cond_obs , noisy_target ]
        - conditional (guide):    noisy_target only (模型内部再用itp_info)
        返回形状 [B,C_in,K,L]
        """
        if self.is_unconditional is True:
            total_input = noisy_data.unsqueeze(1)  # [B,1,K,L]
        else:
            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # [B,2,K,L]
            else:
                total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)  # [B,1,K,L]
        return total_input

    # ---------- 反向扩散采样 ----------

    def reverse_diffusion_sample(
        self,
        observed_data,  # [B,K,L]
        cond_mask,      # [B,K,L]
        side_info,      # [B,C,S,T]
        n_samples,
        itp_info,       # [B,1,K,L] or None
    ):
        """
        走整个反向扩散链，从纯噪声开始一步步还原，得到插补样本。
        返回: [B, n_samples, K, L]
        """
        B, K, L = observed_data.shape
        samples_out = torch.zeros(B, n_samples, K, L, device=self.device)

        for i in range(n_samples):
            # (可选) unconditional 预先生成 noisy_cond_history 以固定条件
            if self.is_unconditional is True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + \
                                self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            # current_sample: 随机初始
            current_sample = torch.randn_like(observed_data)

            # 反向走扩散时间步
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional is True:
                    diff_input = cond_mask * noisy_cond_history[t] + \
                                 (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # [B,1,K,L]
                else:
                    if not self.use_guide:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat(
                            [cond_obs, noisy_target], dim=1
                        )  # [B,2,K,L]
                    else:
                        diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)

                predicted = self.diffmodel(
                    diff_input,
                    side_info,
                    torch.tensor([t], device=self.device),
                    itp_info,
                )

                coeff1 = 1 / (self.alpha_hat[t] ** 0.5)
                coeff2 = (1 - self.alpha_hat[t]) / ((1 - self.alpha[t]) ** 0.5)
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            samples_out[:, i] = current_sample.detach()

        return samples_out

    # ---------- forward / inference ----------

    def forward(self, batch, is_train=1):
        """
        Lightning/训练循环会直接调这个。
        我们期望上层类把 batch 先整理好：
            observed_data: [B,S,T,D] -> [B,K,L,*]
            observed_mask: [B,K,L]
            cond_mask:     [B,K,L]
            coeffs:        [...可能为 None...]
        然后传进来。

        这里做：
        1. 生成 side_info (时空条件嵌入)
        2. 调 loss_single_timestep 或 loss_average_over_schedule
        """

        (
            observed_data,     # [B,K,L,*] 但我们后面只用 [...,0] -> [B,K,L]
            observed_mask,     # [B,K,L]
            coeffs,            # [B,K,L] (after squeeze/permute) or None
            cond_mask,         # [B,K,L]
        ) = self.prepare_batch_train(batch)  # 注意: 上层会传已经处理好的tuple进来

        side_info = self.Temb(observed_data)

        guided_itp = None
        if self.use_guide and coeffs is not None:
            # 原逻辑: coeffs.unsqueeze(1) 变成 [B,1,K,L]
            guided_itp = coeffs.unsqueeze(1)

        # 根据 is_train 选择 loss 函数
        loss_func = self.loss_single_timestep if is_train == 1 else self.loss_average_over_schedule

        loss = loss_func(
            observed_data[:,:,:,0],  # 只取第0特征通道
            cond_mask,
            observed_mask,
            side_info,
            guided_itp,
            is_train,
        )
        return loss

    def inference_step(self, batch, n_samples):
        """
        推理/评估阶段用:
        给一个eval batch -> 生成插补样本、target、mask等
        """
        (
            observed_data,  # [B,K,L,*]
            observed_mask,  # [B,K,L]
            gt_mask,        # [B,K,L]
            cut_length,     # [B] int, 对尾部padding的修正
            coeffs,         # [B,K,L] or None
        ) = self.prepare_batch_eval(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask  # 哪些位置当成 supervised target

            side_info = self.Temb(observed_data)

            guided_itp = None
            if self.use_guide and coeffs is not None:
                guided_itp = coeffs.unsqueeze(1)  # [B,1,K,L]

            # 采样 n 条反向扩散轨迹
            samples = self.reverse_diffusion_sample(
                observed_data[:,:,:,0],
                cond_mask,
                side_info,
                n_samples,
                guided_itp,
            )

            # 避免对 padding/tail 做评估
            for i in range(len(cut_length)):
                target_mask[i, ..., 0:cut_length[i].item()] = 0

        # 返回和 evaluate() 原来类似的四元组
        return samples, observed_data[:,:,:,0], target_mask, observed_mask


class GuSTDiff(GuSTDiffCore):
    """
    这个类是“数据接口适配层”:
    - 把 dataloader 给的一堆键 (observed_data, observed_mask, coeffs, gt_mask, ...)
      变成 core 需要的张量(permute / squeeze 等)
    - 继承 core 的 forward() / inference_step()，供外部Trainer使用
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def prepare_batch_train(self, raw_batch):
        """
        从 dataloader 的 batch 里提取并重排出训练所需的四个张量:
          observed_data: [B,S,T,D] -> permute -> [B,K,L,D]
          observed_mask: [B,K,L]
          coeffs:        [B,K,L] (only if use_guide)
          cond_mask:     [B,K,L]
        """
        observed_data = raw_batch["observed_data"].to(self.device).float()
        observed_mask = raw_batch["observed_mask"].to(self.device).float()
        cond_mask     = raw_batch["cond_mask"].to(self.device).float()

        coeffs = None
        if self.args.use_guide:
            coeffs_raw = raw_batch["coeffs"].to(self.device).float()  # [B,S,T,D?]
            # 你原本的处理: [:,:,:,0].permute(0, 2, 1)
            coeffs = coeffs_raw[:,:,:,0].permute(0, 2, 1)  # -> [B,K,L]

        # permute 到 [B,K,L,D]
        observed_data = observed_data.permute(0, 2, 1, 3)
        # [B,S,T] -> [B,K,L]
        observed_mask = observed_mask.permute(0, 2, 1)
        cond_mask     = cond_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            coeffs,
            cond_mask,
        )

    def prepare_batch_eval(self, raw_batch):
        """
        用于验证/测试:
          observed_data, observed_mask, gt_mask, cut_length, coeffs
        """
        observed_data = raw_batch["observed_data"].to(self.device).float()
        observed_mask = raw_batch["observed_mask"].to(self.device).float()
        gt_mask       = raw_batch["gt_mask"].to(self.device).float()
        cut_length    = raw_batch["cut_length"].to(self.device).long()

        coeffs = None
        if self.args.use_guide:
            coeffs_raw = raw_batch["coeffs"].to(self.device).float()
            coeffs = coeffs_raw[:,:,:,0].permute(0, 2, 1)  # [B,K,L]

        observed_data = observed_data.permute(0, 2, 1, 3)  # [B,K,L,D]
        observed_mask = observed_mask.permute(0, 2, 1)     # [B,K,L]
        gt_mask       = gt_mask.permute(0, 2, 1)           # [B,K,L]

        return (
            observed_data,
            observed_mask,
            gt_mask,
            cut_length,
            coeffs,
        )
