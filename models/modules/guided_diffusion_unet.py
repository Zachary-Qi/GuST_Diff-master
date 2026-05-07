import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .st_diffusion_blocks import *


class GuidedDiffusionUNet(nn.Module):
    """
    条件扩散噪声预测网络（denoiser）。
    - 输入: 当前 noisy 状态 x (以及可选的插值引导 itp_x)
    - 条件: side_info (时空特征), diffusion_step (时间步), itp guidance (如果启用)
    - 输出: 对噪声/残差的估计，用于扩散反推

    结构:
      [输入拼接+1x1 conv] ->
      [可选: itp 引导分支建模] ->
      [SpatioTemporalDenoiserBlock 主干, 带 diffusion step 注入] ->
      [head -> 1x1 conv -> reshape到 [B,K,L]]
    """

    def __init__(self, args, inputdim=1, target_dim=36, is_itp=False, emb_total_dim=64):
        """
        args:
          channels: 主干通道数
          nheads, layers: 注意力/堆叠深度
          diffusion_embedding_dim
          num_steps
          device
        inputdim:
          输入通道数，通常是:
            - 1 (only noisy_target)
            - 2 (cond + noisy_target)
            - 如果 use_guide=True，还会concat itp_x
        target_dim:
          空间维度 (节点数 K)
        is_itp:
          是否使用插值-引导 (guide)
        emb_total_dim:
          时空 side_info 投影后的channel数
        """
        super().__init__()
        self.channels = args.channels
        self.is_itp = is_itp  # True 表示有插值引导
        self.itp_channels = None

        # ====== guidance / itp 分支 ======
        if self.is_itp:
            self.itp_channels = self.channels

            # 把插值引导的 itp_x (inputdim-1 通道)做1x1卷积到 itp_channels
            self.itp_projection = Conv2dKaiming(inputdim - 1, self.channels, 1, 1, 0)

            # 根据 side_info 做条件调制后，再跑一个专门的 modeling 块
            self.itp_modeling = GuidanceFeatureRefiner(
                channels=self.itp_channels,
                nheads=args.nheads,
                layers=args.layers,
                dropout = args.dropout
            )

            # 把 side_info 投影到和 itp_x 相同的通道数，以便融合
            self.cond_projection = Conv2dKaiming(
                emb_total_dim,
                self.itp_channels,
                1, 1, 0
            )

        # ====== diffusion step embedding ======
        self.diffusion_embedding = DiffusionStepEmbedding(
            num_steps=args.num_steps,
            embedding_dim=args.diffusion_embedding_dim,
            spatial_dim=target_dim,
        )

        self.device = args.device

        # ====== 输入投影 ======
        # 把 (noisy target [+ cond] [+ itp]) 映射到主干通道数
        self.inp_proj = Conv2dKaiming(inputdim, self.channels, 1, 1, 0)

        # ====== 时空主干 (多层注意力 + 残差门控) ======
        self.st_backbone = SpatioTemporalDenoiserBlock(
            side_dim=emb_total_dim,
            channels=self.channels,
            diffusion_embedding_dim=args.diffusion_embedding_dim,
            nheads=args.nheads,
            target_dim=target_dim,
            device=args.device,
            layers=args.layers,
            dropout = args.dropout
        )

        # ====== 输出头 ======
        # 先 1x1 conv 到 channels，再relu
        self.mid_proj = Conv2dKaiming(self.channels, self.channels, 1, 1, 0)

        # 最终线性到 1 通道 (每个node每个时间步一个标量噪声预测)
        self.out_proj = conv1d_kaiming(self.channels, 1, 1)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x, side_info, diffusion_step, itp_x):
        """
        x:      [B, C_in, K, L] 当前 noisy 输入 (可能是 noisy_target 或 concat(cond,noisy_target))
        side_info: [B, side_dim, K, L] 时空辅助特征 (来自 TimeFeatsEmbed 等)
        diffusion_step: [B] or [B,1] 当前扩散时间步 t
        itp_x:  [B, ?, K, L]  (仅在 is_itp=True 时需要)，插值引导轨迹

        return:
          pred_noise: [B, K, L]
        """
        # 如果是引导版，把 itp_x 拼进 x 的通道尾部
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)

        B, inputdim, K, L = x.shape

        # 1) 输入 1x1 proj
        x = self.inp_proj(x)      # -> [B, channels, K, L]
        x = F.relu(x)

        # 2) 如果是引导变体, 先对 itp_x 做条件化建模
        if self.is_itp:
            # (a) 先把 itp_x 单独做1x1投影
            itp_x = self.itp_projection(itp_x)  # [B, channels, K, L]

            # (b) side_info -> cond_embedding，叠加到 itp_x
            itp_cond_info = self.cond_projection(side_info)  # [B, channels, K, L]
            itp_x = itp_x + itp_cond_info

            # (c) 引导分支的 attention/transformer-like 模块
            itp_x = self.itp_modeling(itp_x)    # [B, channels, K, L]
            itp_x = F.relu(itp_x)
        else:
            # 如果不使用引导，就把 itp_x 留空/None，主干里会自己处理
            itp_x = itp_x  # keep for API consistency

        # 3) diffusion step embedding -> 时空主干
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        # st_backbone 负责时空注意力 + 残差门控融合
        x = self.st_backbone(x, side_info, diffusion_emb, itp_x)

        # 4) 输出 head
        x = self.mid_proj(x)    # [B, channels, K, L]
        x = F.relu(x)

        # reshape 给 out_proj (Conv1d)
        x = x.reshape(B, self.channels, K * L)  # [B, C, K*L]
        x = self.out_proj(x)                    # [B, 1, K*L]

        # 回 reshape 成 [B,K,L]
        x = x.reshape(B, K, L)

        return x


class SpatioTemporalDenoiserBlock(nn.Module):
    """
    时空扩散残差块:
    - 将 diffusion 时间步嵌入注入到特征
    - 多层 temporal + spatial attention
    - 融合 side_info 条件
    - gated residual 输出 (WaveNet式门控+skip分离)

    这个模块就是你原本的 NoiseProject，只是命名和内部变量更语义化。
    """

    def __init__(
        self,
        side_dim,
        channels,
        diffusion_embedding_dim,
        nheads,
        target_dim,
        device=None,
        layers=4,
        dropout=0,
    ):
        super().__init__()

        # 把扩散时间步 embedding (per-node) 映射到通道维
        self.diffusion_step_proj = nn.Linear(diffusion_embedding_dim, channels)

        # 用于融合 side_info 到主干特征的投影
        self.cond_side_proj = Conv2dKaiming(side_dim, 2 * channels, 1, 1, 0)

        # 把时空注意力后的特征映射到 2*channels (用于门控)
        self.fusion_proj = Conv2dKaiming(channels, 2 * channels, 1, 1, 0)

        # 把门控之后的输出再次投影到 2*channels，并进行 residual/skip split
        self.residual_gate_proj = Conv2dKaiming(channels, 2 * channels, 1, 1, 0)

        # 多层 temporal / spatial attention block 堆叠
        self.temporal_attn_layers = nn.ModuleList([
            TemporalCrossProjectionBlock(
                target_dim,      # seq_len / 时间方向投影长度? （你的实现里叫 target_dim，实为 K?）
                10,              # dim_proj=10 (你原来硬编码的)
                channels,
                nheads,
                channels,
                dropout
            )
            for _ in range(layers)
        ])

        self.spatial_attn_layers = nn.ModuleList([
            SpatialGuidedAttentionBlock(
                device,
                channels,
                channels,
                nheads,
                channels * 2
            )
            for _ in range(layers)
        ])

    def forward(self, x, side_info, diffusion_emb, itp_info):
        """
        x:           [B, C, K, L] 主特征 (来自输入投影/上层residual)
        side_info:   [B, side_dim, K, L] 条件时空embedding
        diffusion_emb: 输出形状来自 DiffusionStepEmbedding(t)，通常 [B, K, diffusion_dim]
        itp_info:    [B, C, K, L] (若使用引导) or None

        return:
          x_res:     [B, C, K, L] 经过时空注意力+门控后的残差输出
        """
        # 1) 注入扩散时间步 embedding
        # diffusion_emb: [B, K, diffusion_dim]
        # diffusion_step_proj -> [B, K, C]
        diffusion_emb = self.diffusion_step_proj(diffusion_emb)  # [B, K, C]
        # reshape成 [B, C, K, 1] 再加到 x
        diffusion_emb = diffusion_emb.unsqueeze(-1).permute(0, 2, 1, 3)
        # 现在 diffusion_emb shape: [B, C, K, 1]

        # y: [B, K, L, C]  (把 x+diffusion_emb 先加，再换维给注意力)
        y = (x + diffusion_emb).permute(0, 2, 3, 1)

        # 2) 堆叠 temporal / spatial attention
        #    temporal_attn_layers: 对时间展开
        #    spatial_attn_layers:  对空间用额外引导 (itp_info)
        for att_t, att_s in zip(self.temporal_attn_layers, self.spatial_attn_layers):
            y = att_t(y)  # 期望输出: [B, K, L, C]
            # itp_info: [B, C, K, L] -> permute成 [B, L, K, C] 再作为 emb?
            y = att_s(
                y,
                itp_info.permute(0, 3, 2, 1),  # [B, L, K, C]  (保持你原来的行为)
                dim=1
            )
            # y 仍是 [B, K, L, C]

        # 回到 [B, C, K, L]
        y = y.permute(0, 3, 1, 2)

        # 3) 把时空注意力输出过 projection (fusion_proj) 得到 [B, 2C, K, L]
        y = self.fusion_proj(y)

        # 把 side_info 也投影到 [B, 2C, K, L]，与 y 融合
        side_info = self.cond_side_proj(side_info)

        y = y + side_info  # 条件注入

        # 4) Gated activation 单元 (WaveNet风格门控)
        gate, filt = torch.chunk(y, 2, dim=1)  # [B,C,K,L] x2
        y = torch.sigmoid(gate) * torch.tanh(filt)

        # 再过 residual_gate_proj -> [B, 2C, K, L]
        y = self.residual_gate_proj(y)

        # residual & skip 分离
        residual, skip = torch.chunk(y, 2, dim=1)  # [B,C,K,L] each

        # 最后做残差融合，尺度归一
        return (x + residual) / math.sqrt(2.0)
