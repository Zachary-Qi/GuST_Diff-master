import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import torch.nn.init as init


def conv1d_kaiming(in_channels, out_channels, kernel_size):
    """
    Helper: 创建 Conv1d 并用 kaiming_normal_ 初始化权重。
    """
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionStepEmbedding(nn.Module):
    """
    把扩散步骤 t (0..num_steps-1) 映射到一个连续向量表示，并做两层 MLP (SiLU)，
    然后 broadcast 到每个空间点 (spatial_dim)。

    输出形状: [B, spatial_dim, projection_dim]
    """
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None, spatial_dim=207):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        self.spatial_dim = spatial_dim

        # 预计算的正弦-余弦时间步 embedding 表
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )

        # 小 MLP
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        """
        diffusion_step: LongTensor [B], 表示 t
        """
        # 查时间步嵌入
        x = self.embedding[diffusion_step]  # [B, embedding_dim]

        # 两层非线性投影
        x = F.silu(self.proj1(x))
        x = F.silu(self.proj2(x))           # [B, projection_dim]

        # broadcast 到每个空间位置 (sensor/node)
        x = x.unsqueeze(1)                  # [B, 1, projection_dim]
        x = x.expand(-1, self.spatial_dim, -1)  # [B, spatial_dim, projection_dim]
        return x

    def _build_embedding(self, num_steps, dim=64):
        """
        经典扩散/Transformer风格的时间步正弦余弦嵌入:
            sin( t * freq ), cos( t * freq )
        """
        steps = torch.arange(num_steps).unsqueeze(1)  # [T,1]
        # 指数式频率范围 (10^(...))
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # [1,dim]
        table = steps * frequencies  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [T,2*dim]
        return table


class Conv2dKaiming(nn.Module):
    """
    一个带 kaiming_normal_ 初始化的 2D 卷积层封装。
    只有卷积本身，不带 BN/激活（由外层控制）。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class MultiHeadAttentionBlock(nn.Module):
    """
    基础多头注意力(Q/K/V线性投影)，不包含 FFN / 残差 / Norm。
    它在指定的倒数第二维上做attention (tgt_length/src_length)，最后一维是 model_dim。

    输入:
      query: [B, ..., Tq, D]
      key:   [B, ..., Tk, D]
      value: [B, ..., Tk, D]

    输出:
      out:   [B, ..., Tq, D]
    """
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        B = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        # 做Q,K,V投影
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 拆多头: concat(heads) along batch dim
        Q = torch.cat(torch.split(Q, self.head_dim, dim=-1), dim=0)
        K = torch.cat(torch.split(K, self.head_dim, dim=-1), dim=0)
        V = torch.cat(torch.split(V, self.head_dim, dim=-1), dim=0)

        # 注意力打分 (QK^T / sqrt(d))
        K = K.transpose(-1, -2)  # [B*H, ..., head_dim, src_length]
        attn_score = (Q @ K) / (self.head_dim ** 0.5)  # [B*H, ..., tgt_length, src_length]

        # causal mask可选
        if self.mask:
            mask = torch.ones(
                tgt_length,
                src_length,
                dtype=torch.bool,
                device=Q.device
            ).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)

        # 乘 V
        out = attn_score @ V  # [B*H, ..., tgt_length, head_dim]

        # 合并回原 batch 维
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)  # [B, ..., tgt_length, D]

        # 最终线性
        out = self.out_proj(out)
        return out


class MoEFFN(nn.Module):
    """
    MoE FFN: x -> sum_{e in topk} p_e * Expert_e(x)
    输入:  x [..., D]
    输出:  y [..., D]
    """
    def __init__(self, model_dim, feed_forward_dim, num_experts=4, k=2, dropout=0.0):
        super().__init__()
        assert k <= num_experts
        self.num_experts = num_experts
        self.k = k

        # router/gating
        self.router = nn.Linear(model_dim, num_experts)

        # experts: 用你原来的两层FFN结构
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, feed_forward_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feed_forward_dim, model_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [*, D]
        orig_shape = x.shape
        D = orig_shape[-1]
        x_flat = x.reshape(-1, D)  # [N, D]

        logits = self.router(x_flat)               # [N, E]
        topv, topi = logits.topk(self.k, dim=-1)   # [N, k]
        topw = F.softmax(topv, dim=-1)             # [N, k]

        # 计算所有 expert 输出（简单但显存大；E不大时OK）
        expert_out = torch.stack([e(x_flat) for e in self.experts], dim=1)  # [N, E, D]

        # gather top-k experts: [N, k, D]
        idx = topi.unsqueeze(-1).expand(-1, -1, D)
        chosen = torch.gather(expert_out, dim=1, index=idx)

        # 加权求和: [N, D]
        y = (chosen * topw.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)


class ResidualSelfAttentionBlock_MoE(nn.Module):
    """
    Self-Attn + MoE-FFN with residual + LayerNorm
    """
    def __init__(self,
                 model_dim,
                 feed_forward_dim=2048,
                 num_heads=8,
                 dropout=0.0,
                 mask=False,
                 num_experts=4,
                 topk=2):
        super().__init__()

        self.self_attn = MultiHeadAttentionBlock(model_dim, num_heads, mask)

        # 用 MoE 替代原 FFN
        self.moe_ffn = MoEFFN(
            model_dim=model_dim,
            feed_forward_dim=feed_forward_dim,
            num_experts=num_experts,
            k=topk,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)  # [..., L, D]
        residual = x

        # Self-attn
        out = self.self_attn(x, x, x)
        out = self.drop1(out)
        out = self.norm1(residual + out)

        # MoE-FFN
        residual = out
        out2 = self.moe_ffn(out)      # [..., L, D]
        out2 = self.drop2(out2)
        out = self.norm2(residual + out2)

        out = out.transpose(dim, -2)
        return out



class ResidualSelfAttentionBlock(nn.Module):
    """
    一个标准Transformer-style块 (Self-Attention + FFN)，带残差+LayerNorm。
    - 支持指定 dim 参数，表示在哪个维度上视作 "sequence length"。
      我们会先 transpose(dim, -2)，让 sequence 轴到倒数第二维。
    """
    def __init__(self,
                 model_dim,
                 feed_forward_dim=2048,
                 num_heads=8,
                 dropout=0,
                 mask=False):
        super().__init__()

        self.self_attn = MultiHeadAttentionBlock(model_dim, num_heads, mask)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        """
        x: [B, ..., L, D]  (如果 dim != -2，会transpose过来做注意力)
        """
        x = x.transpose(dim, -2)  # -> [..., L, D] on -2
        residual = x

        # Self-attn (Q=K=V=x)
        out = self.self_attn(x, x, x)
        out = self.drop1(out)
        out = self.norm1(residual + out)

        residual = out
        out = self.ffn(out)
        out = self.drop2(out)
        out = self.norm2(residual + out)

        out = out.transpose(dim, -2)
        return out


class GuidanceFeatureRefiner(nn.Module):
    """
    用于引导分支(itp_x)的特征精炼:
    - 堆叠多层时间注意力+空间注意力 (ResidualSelfAttentionBlock)
    - 简单的两层前馈扩展/压缩
    - 残差 + GroupNorm

    输入 y: [B, C, K, L]
    输出:   [B, C, K, L] (同shape)
    """
    def __init__(self, channels, nheads, layers, dropout):
        super().__init__()

        # 多层 [时间注意力块, 空间注意力块] 交替
        self.temporal_blocks = nn.ModuleList([
            ResidualSelfAttentionBlock(
                model_dim=channels,
                feed_forward_dim=channels * 2,
                num_heads=nheads,
                dropout=dropout,
                mask=False,
            )
            for _ in range(layers)
        ])
        # self.temporal_blocks = nn.ModuleList([
        #     ResidualSelfAttentionBlock_MoE(
        #         model_dim=channels,
        #         feed_forward_dim=channels * 2,
        #         num_heads=nheads,
        #         dropout=dropout,
        #         num_experts=6,
        #         topk=2
        #     )
        #     for _ in range(layers)
        # ])


        self.spatial_blocks = nn.ModuleList([
            ResidualSelfAttentionBlock(
                model_dim=channels,
                feed_forward_dim=channels * 2,
                num_heads=nheads,
                dropout=dropout,
                mask=False,
            )
            for _ in range(layers)
        ])
        # self.spatial_blocks = nn.ModuleList([
        #     ResidualSelfAttentionBlock_MoE(
        #         model_dim=channels,
        #         feed_forward_dim=channels * 2,
        #         num_heads=nheads,
        #         dropout=dropout,
        #         num_experts=6,
        #         topk=2
        #     )
        #     for _ in range(layers)
        # ])

        # 小两层FFN: C -> 2C -> C
        # self.ffn_up = nn.Linear(channels, 128)
        # self.ffn_down = nn.Linear(128, channels)
        self.ffn_up = nn.Linear(channels, channels * 2)
        self.ffn_down = nn.Linear(channels * 2, channels)

        self.out_norm = nn.GroupNorm(4, channels)
        # self.gamma = nn.Parameter(torch.tensor(0.1))  # 学习一个缩放系数

    def forward(self, y):
        """
        y: [B, C, K, L]
        返回: 同shape
        """
        y_in = y  # for final residual

        # 注意: ResidualSelfAttentionBlock 期望 [..., length, dim_model]
        # 这里我们把 y 转成 [B, L, K, C]，让 dim=1 当成时间，dim=2 当成空间
        y_work = y.permute(0, 3, 2, 1)  # [B, L, K, C]

        # 时序注意力 + 空间注意力 交替
        for block_t, block_s in zip(self.temporal_blocks, self.spatial_blocks):
            # 时间方向: 把 dim=1 当作 sequence 轴
            y_work = block_t(y_work, dim=1)
            # 空间方向: 把 dim=2 当作 sequence 轴
            y_work = block_s(y_work, dim=2)

        # FFN on channel dim
        y_ff = F.relu(self.ffn_up(y_work))
        y_ff = self.ffn_down(y_ff)  # [B, L, K, C]

        # 回到 [B, C, K, L]
        out = y_in + y_ff.permute(0, 3, 2, 1)
        
        # out = y_in + self.gamma * y_ff.permute(0, 3, 2, 1)
        
        out = self.out_norm(out)
        return out


class TemporalCrossProjectionBlock(nn.Module):
    """
    时间方向的'低秩锚点交互'注意力块:
    - anchor_tokens: 可学习的 dim_proj × d_model (相当于一组低维锚点/摘要 token)
    - anchor_to_token_attn: 让 anchor 从序列 x 里读信息
    - token_to_anchor_attn: 让 x 从 anchor 里回收融合后的上下文
    - FFN + 残差 + Norm

    输入:
      x: [B, S, N, D]  (比如 S=seq_len, N=num_nodes)
    输出:
      same shape as x
    """
    def __init__(self, seq_len, dim_proj, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.anchor_to_token_attn = MultiHeadAttentionBlock(d_model, n_heads, mask=None)
        self.token_to_anchor_attn = MultiHeadAttentionBlock(d_model, n_heads, mask=None)

        # 可学习的 "锚点 tokens"
        self.anchor_tokens = nn.Parameter(torch.randn(dim_proj, d_model))

        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.seq_len = seq_len

    def forward(self, x):
        """
        x: [B, S, N, D]
        """
        B = x.shape[0]

        # broadcast anchor_tokens 到 batch & seq_len 维度:
        # anchor_tokens: [dim_proj, D]
        # -> [B, S, dim_proj, D]
        anchors = repeat(
            self.anchor_tokens,
            'dim_proj d_model -> repeat seq_len dim_proj d_model',
            repeat=B,
            seq_len=self.seq_len
        )  # [B, S, C, D]  这里C=dim_proj

        # 1. anchors 从 x 读取信息
        #    anchors 作为 Q; x 作为 K,V
        msg_from_x = self.anchor_to_token_attn(
            anchors,  # Q
            x,        # K
            x         # V
        )  # [B, S, C, D]

        # 2. x 再从 anchors 读取融合后的信息
        #    x 作为 Q; msg_from_x 作为 K,V
        msg_to_x = self.token_to_anchor_attn(
            x,             # Q
            msg_from_x,    # K
            msg_from_x     # V
        )  # [B, S, N, D]

        # 残差 + norm + FFN + 残差 + norm
        h = self.norm1(x + self.drop(msg_to_x))
        h2 = self.ffn(h)
        h2 = self.norm2(h + self.drop(h2))

        return h2


class SpatialGuidedAttention(nn.Module):
    """
    空间引导注意力:
    - emb 用来生成 Q/K
    - value 用来生成 V
    - 多头注意力后 cat 回 batch, 再线性映射回 model_dim

    输入:
      value: [B, ..., L, D]  (原特征)
      emb:   [B, ..., L, D_emb] (引导嵌入/自适应embedding)
    输出:
      out:   [B, ..., L, D]
    """
    def __init__(self, model_dim, adaptive_embedding_dim, nheads, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = nheads
        self.mask = mask
        self.head_dim = model_dim // nheads

        # Q,K 从 emb 来
        self.qk_proj = nn.Linear(adaptive_embedding_dim, model_dim)
        # V 从 value 来
        self.v_proj = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, value, emb):
        # batch size
        B = value.shape[0]
        tgt_length = value.shape[-2]
        src_length = value.shape[-2]

        # Q,K from emb
        Q = self.qk_proj(emb)
        K = self.qk_proj(emb)
        # V from value
        V = self.v_proj(value)

        # split heads by cat along batch dim
        Q = torch.cat(torch.split(Q, self.head_dim, dim=-1), dim=0)
        K = torch.cat(torch.split(K, self.head_dim, dim=-1), dim=0)
        V = torch.cat(torch.split(V, self.head_dim, dim=-1), dim=0)

        # attn
        K = K.transpose(-1, -2)   # [B*H, ..., head_dim, src_length]
        attn_score = (Q @ K) / (self.head_dim ** 0.5)

        if self.mask:
            mask = torch.ones(
                tgt_length,
                src_length,
                dtype=torch.bool,
                device=Q.device
            ).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ V  # [B*H, ..., tgt_length, head_dim]

        # merge heads
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)  # [B, ..., tgt_length, D]

        out = self.out_proj(out)
        return out


class SpatialGuidedAttentionBlock(nn.Module):
    """
    空间引导注意力块 (类似 Transformer block):
    - SpatialGuidedAttention (Q,K由外部emb提供)
    - 残差 + norm
    - FFN + 残差 + norm
    """
    def __init__(self,
                 device,
                 model_dim,
                 adaptive_embedding_dim,
                 nheads,
                 feed_forward_dim=2048,
                 dropout=0.1):
        super().__init__()

        self.attn = SpatialGuidedAttention(model_dim, adaptive_embedding_dim, nheads)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, emb, dim=-2):
        """
        x:   [B, ..., L, D]   (feature)
        emb: [B, ..., L, D_emb] (guidance embedding for Q/K)
        """
        x = x.transpose(dim, -2)
        residual = x

        out = self.attn(x, emb)     # guided attention
        out = self.drop1(out)
        out = self.norm1(residual + out)

        residual = out
        out = self.ffn(out)
        out = self.drop2(out)
        out = self.norm2(residual + out)

        out = out.transpose(dim, -2)
        return out
