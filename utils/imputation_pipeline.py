import numpy as np
import pandas as pd
from copy import deepcopy


# 返回形状为 `shape` 的0/1数组；1表示该位置将被置为“缺失/故障”（需要模型去补全）
def sample_mask(shape, p=0.0015, p_noise=0.05, min_seq=1, max_seq=1, rng=None):
    
    """
    生成时空缺失掩码（uint8；1=缺失，0=正常）。

    参数：
        shape: (T, S)
        p:      连续缺失“起点”的基础概率（后续会扩张为长度∈[min_seq, max_seq) 的块）
        p_noise:叠加的点状缺失概率（独立伯努利）
        max_seq/min_seq: 连续缺失块的长度范围
        rng:    numpy 随机数发生器；None 则用全局 np.random

    返回：
        mask: np.uint8, shape=(T,S)，1 表示该位置将被置为缺失/故障
    """
    
    # 支持传入 numpy 随机数发生器 rng（np.random.default_rng）；若为空就用全局np.random
    if rng is None:
        rand = np.random.random      # 生成[0,1)均匀分布
        randint = np.random.randint  # 生成整数
    else:
        rand = rng.random
        randint = rng.integers
    
    # 先按独立伯努利分布采样基础“故障起点”掩码（概率 p）
    mask = rand(shape) < p
    
    # 对每一列扩展为连续故障片段（模拟传感器连续掉线）
    for col in range(mask.shape[1]):
        # 找到该列被采为起点的位置索引
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
            
        # 随机/指定连续片段长度（[min_seq, max_seq)）
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        
        # 将每个起点向后扩展 fault_len 个时间步，形成块状缺失
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        
        # 去重，并限制在合法时间索引范围内
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        
        # 标记这些时间步为缺失
        mask[idxs, col] = True
    
    # 叠加少量“随机噪声型缺失”（概率 p_noise），使模式更接近真实世界
    mask = mask | (rand(mask.shape) < p_noise)
    
    # 返回 0/1（uint8）掩码；1表示该位置为“缺失/故障”
    return mask.astype('uint8')


# def apply_missing_mask(args, ob_data, obj_adj, p_fault, p_noise, mean, std, min_seq=12, max_seq=12*4, seed=66666,
#                        *,
#                        use_nan_as_missing: bool = False,
#                        return_numpy: bool = True,):
#     """
#     作用：
#       1) 基于原始观测生成 obj_mask（1=原始有观测；0=原始无观测）
#       2) 采样 eval_mask（1=人为缺失），显式制造“遮挡-重建”的训练/评估目标
#       3) 计算 gt_mask = 1 - (eval_mask | (1 - obj_mask))，保证“只在有真值处监督/评估”
#       4) 对原始数据做标准化（减均值/除标准差），并在原始无观测处清零 => c_data
#       5) 按 train/val/test 比例切分，返回三段的 observed_data / observed_mask / gt_mask

#     评估协议说明：
#       - 训练阶段：在 Dataset 内部通过 cond_mask（可见性）构造“可见子集”，形成掩码自监督；
#       - 验证/测试默认使用 gt_mask 作为 cond_mask，确保指标仅统计“可监督真值”位置（公平评估）；
#       - 若需要横向可比的点状缺失基准，可在测试阶段额外指定 missing_ratio，并采用 point 协议遮挡。

#     参数：
#         args: 需要包含 train_ratio / val_ratio ∈ (0,1)，且二者之和 < 1
#         obj_data: 原始数据 DataFrame，形状 (T,S)
#         obj_adj: 保留参数
#         p_fault/p_noise/min_seq/max_seq/seed: 缺失采样超参
#         mean/std: 标准化用（标量或 shape=[S]）
#         use_nan_as_missing: True 则以 NaN 判定“原始无观测”；False 则以 0 判定（与一些数据预处理兼容）
#         return_numpy: True 返回 np.ndarray；False 返回带 index/columns 的 DataFrame

#     返回：
#         {
#           "train": {"observed_data","observed_mask","gt_mask"},
#           "val":   {...},
#           "test":  {...},
#           "meta":  {"eval_mask","obj_mask","mean","std","index","columns"}
#         }
#     """
    
#     # 固定随机种子，使用新的生成器（推荐方式）
#     random = np.random.default_rng(seed)
    
#     T, S, D = ob_data.shape
#     obj_data = ob_data[..., 0]
#     # 原始观测掩码：数据不为0的位置视为“有观测”（1），为0视为“无观测”（0）
#     # 注：这假设0不是有效值；若0是合法数值，这里需要改成基于NaN或单独的原始掩码
#     obj_mask = (obj_data.values != 0.).astype('uint8')
    
#     # 原始观测掩码（1=原始有观测）
#     if use_nan_as_missing:
#         obj_mask = (~obj_data.isna()).values.astype("uint8")
#         data_filled = obj_data.fillna(0).values
#     else:
#         obj_mask = (obj_data.values != 0.0).astype("uint8")
#         data_filled = obj_data.fillna(0).values
    
#     # 评估用缺失掩码：在原数据上“人为制造缺失”，用于评估插补性能
#     # 1 表示我们要抹掉（变缺失）的点
#     eval_mask = sample_mask(shape=(T, S), p=p_fault, p_noise=p_noise, min_seq=min_seq, max_seq=max_seq, rng=random)
    
#     # 计算“可用于评估的真值掩码” gt_mask（1表示此处既有原始观测，又未被eval_mask抹掉）
#     # 逻辑：eval_mask=1（人为缺失） 或 原本就无观测(1-obj_mask=1) 的位置，都不能当作GT
#     # 因此用 1 - (eval_mask | (1-obj_mask)) 得到“可做GT”的位置
#     gt_mask = (1 - (eval_mask | (1-obj_mask))).astype('uint8')
    

#     # 对数据做标准化（减均值/除标准差），再与原始观测掩码相乘（把未观测位置清0）
#     # 若 mean/std 为标量或按列的ndarray都可；fillna(0) 先把NaN补为0再标准化
#     mean = np.asarray(mean).reshape(1, -1)
#     std = np.asarray(std).reshape(1, -1)
#     std = np.where(std == 0, 1.0, std)
    
#     c_data = (
#              (data_filled - mean) / std
#         ) * obj_mask
#     print("c_data:", c_data.shape)
#     # 数据集切分：训练/验证/测试的边界（下标）
#     train_end = int(args.train_ratio * T)
#     val_end = int((args.train_ratio + args.val_ratio) * T)
    
#     splits = {
#         "train": slice(0, train_end),
#         "val":   slice(train_end, val_end),
#         "test":  slice(val_end, T),
#     }
    
#     def _maybe_df(arr: np.ndarray):
#         if return_numpy:
#             return arr
#         return pd.DataFrame(arr, index=obj_data.index, columns=obj_data.columns)

#     out = {}
#     for name, sl in splits.items():
#         out[name] = {
#             "observed_data": _maybe_df(c_data[sl].astype(np.float32)),
#             "observed_mask": _maybe_df(obj_mask[sl].astype(np.uint8)),
#             "gt_mask":       _maybe_df(gt_mask[sl].astype(np.uint8)),
#         }

#     out["meta"] = {
#         "eval_mask": _maybe_df(eval_mask.astype(np.uint8)),
#         "obj_mask":  _maybe_df(obj_mask.astype(np.uint8)),
#         "mean": c_data[:1].astype(np.float32).reshape(-1) * 0 + mean.astype(np.float32).reshape(-1),  # just pack
#         "std":  c_data[:1].astype(np.float32).reshape(-1) * 0 + std.astype(np.float32).reshape(-1),
#         "index": None if return_numpy else obj_data.index,
#         "columns": None if return_numpy else obj_data.columns,
#     }
#     return out


def apply_missing_mask(
    args,
    ob_data: np.ndarray,           # (T,S,D): 0通道=数值，其他通道=时间等特征
    p_fault, p_noise,
    mean, std,                     # 仅用于 0 通道：标量 或 (S,)
    min_seq=12, max_seq=12*4, seed=66666,
    *,
    use_nan_as_missing: bool = False,
    return_numpy: bool = True,
):
    """
    输出：
      - train/val/test: {
            "observed_data": (T',S,D),  # 0通道标准化并在原始缺失处清零；其他通道原样保留
            "observed_mask": (T',S),    # 0通道的原始观测掩码
            "gt_mask":       (T',S),    # 真值监督掩码
        }
      - meta: {"eval_mask","obj_mask","mean","std"}（都基于 0 通道的 2D）
    """
    if not isinstance(ob_data, np.ndarray) or ob_data.ndim != 3:
        raise ValueError(f"ob_data 必须是 3D ndarray (T,S,D)，当前类型/形状={type(ob_data)}/{getattr(ob_data,'shape',None)}")

    rng = np.random.default_rng(seed)

    T, S, D = ob_data.shape
    if D < 1:
        raise ValueError("最后一维 D 至少为 1（第 0 通道为数值通道）")

    # 第0通道（值）
    val0 = ob_data[..., 0]  # (T,S)

    # 原始观测掩码 & 填充
    if use_nan_as_missing:
        obj_mask = (~np.isnan(val0)).astype('uint8')          # (T,S)
        X_filled = np.nan_to_num(ob_data, nan=0.0)            # (T,S,D)
    else:
        obj_mask = (val0 != 0.0).astype('uint8')              # (T,S)
        X_filled = np.where(np.isnan(ob_data), 0.0, ob_data)  # (T,S,D)

    # 人为缺失 (2D)
    eval_mask = sample_mask(
        shape=(T, S),
        p=p_fault, p_noise=p_noise,
        min_seq=min_seq, max_seq=max_seq,
        rng=rng
    ).astype('uint8')  # 1=人为缺失

    # 真值监督掩码 (2D)
    gt_mask = (1 - (eval_mask | (1 - obj_mask))).astype('uint8')

    # 标准化仅作用于 0 通道
    if mean is not None and std is not None:
        mean = np.asarray(mean)
        std  = np.asarray(std)
        std  = np.where(std == 0, 1.0, std)

        # 支持 标量 或 (S,)
        if mean.ndim == 0:
            mean0 = mean.reshape(1, 1)       # (1,1)
        elif mean.shape == (S,):
            mean0 = mean.reshape(1, S)       # (1,S)
        else:
            raise ValueError(f"mean 需为标量或 (S,)，当前 {mean.shape}")

        if std.ndim == 0:
            std0 = std.reshape(1, 1)
        elif std.shape == (S,):
            std0 = std.reshape(1, S)
        else:
            raise ValueError(f"std 需为标量或 (S,)，当前 {std.shape}")

        val0_std = (X_filled[..., 0] - mean0) / std0  # (T,S)
    else:
        val0_std = X_filled[..., 0]

    # 仅在原始“无观测”的地方清 0（只作用 0 通道）
    val0_std = (val0_std * obj_mask).astype(np.float32)

    # 组回 3D observed_data
    observed_data_all = X_filled.astype(np.float32).copy()  # (T,S,D)
    observed_data_all[..., 0] = val0_std                    # 覆盖第0通道

    # 切分
    train_end = int(args.train_ratio * T) # T是样本总量 args.train_ratio=0.7
    val_end   = int((args.train_ratio + args.val_ratio) * T) # args.val_ratio=0.1
    splits = {
        "train": slice(0, train_end),
        "val":   slice(train_end, val_end),
        "test":  slice(val_end, T),
    }

    # 构造输出
    out = {}
    for name, sl in splits.items():
        out[name] = {
            "observed_data": observed_data_all[sl],         # (T',S,D)
            "observed_mask": obj_mask[sl].astype(np.uint8), # (T',S)
            "gt_mask":       gt_mask[sl].astype(np.uint8),  # (T',S)
        }

    out["meta"] = {
        "eval_mask": eval_mask.astype(np.uint8),            # (T,S)
        "obj_mask":  obj_mask.astype(np.uint8),             # (T,S)
        "mean": None if mean is None else np.asarray(mean),
        "std":  None if std  is None else np.asarray(std),
    }

    # 仅在你真的需要 DataFrame 时再扩展支持；目前 ndarray 更适配 3D。
    if not return_numpy:
        raise ValueError("当前实现仅返回 NumPy 数组。若需 DataFrame，请先改为 2D 再转换。")

    return out


# def sample_mask(shape, p=0.0015, p_noise=0.05, min_seq=1, max_seq=1, rng=None):
#     """
#     采样二维 (T,S) 缺失掩码：uint8；1=缺失，0=正常
#     用于“值通道”的人为连续块 + 点状缺失
#     """
#     if rng is None:
#         rand = np.random.random
#         randint = np.random.randint
#     else:
#         rand = rng.random
#         randint = rng.integers

#     mask = rand(shape) < p
#     for col in range(shape[1]):
#         idxs = np.flatnonzero(mask[:, col])
#         if not len(idxs):
#             continue
#         fault_len = min_seq
#         if max_seq > min_seq:
#             fault_len += int(randint(max_seq - min_seq))
#         ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
#         ext = np.unique(ext)
#         ext = np.clip(ext, 0, shape[0] - 1)
#         mask[ext, col] = True

#     mask = mask | (rand(shape) < p_noise)
#     return mask.astype('uint8')


# def apply_missing_mask_3d_all_outputs_3d(
#     args,
#     obj_data: np.ndarray,          # 形状 (T,S,D)：D>=1，0通道=数值，1..=时间编码(如tod/dow或其sin/cos)
#     obj_adj,                       # 保留参数
#     p_fault: float, p_noise: float,
#     mean, std,                     # 仅用于 0 通道：标量 或 (S,)
#     min_seq=12, max_seq=12*4, seed=66666,
#     *,
#     use_nan_as_missing: bool = False,
# ):
#     """
#     返回的三元组（train/val/test）中：
#       - observed_data : (T',S,D) —— 0通道做标准化并在“原始无观测处=0”；时间通道保持原样（不隐藏）
#       - observed_mask : (T',S,D) —— 0通道是2D原始掩码扩成3D；时间通道恒为1（可见/可用）
#       - gt_mask       : (T',S,D) —— 0通道=2D真值掩码扩成3D；时间通道恒为0（不参与监督）

#     meta 中也提供 3D 的 obj_mask / eval_mask / gt_mask（按上面规则扩展）。
#     """
#     rng = np.random.default_rng(seed)

#     # ---- 检查形状 ----
#     X = np.asarray(obj_data)
#     if X.ndim != 3:
#         raise ValueError(f"必须是 3D (T,S,D)，当前 {X.shape}")
#     T, S, D = X.shape
#     if D < 1:
#         raise ValueError("最后一维 D 至少为 1（第 0 通道为数值通道）。")

#     # ---- 基于第0通道构造 2D 原始掩码 obj_mask_2d 与填充数据 ----
#     val0 = X[..., 0]  # (T,S)
#     if use_nan_as_missing:
#         obj_mask_2d = (~np.isnan(val0)).astype('uint8')
#         X_filled = np.nan_to_num(X, nan=0.0)     # 全通道 NaN -> 0 仅为数值安全
#     else:
#         obj_mask_2d = (val0 != 0.0).astype('uint8')
#         X_filled = np.where(np.isnan(X), 0.0, X)

#     # ---- 评估用缺失 eval_mask_2d 与 gt_mask_2d（都为 2D）----
#     eval_mask_2d = sample_mask(
#         shape=(T, S),
#         p=p_fault, p_noise=p_noise,
#         min_seq=min_seq, max_seq=max_seq,
#         rng=rng
#     )  # 1=人为缺失
#     gt_mask_2d = (1 - (eval_mask_2d | (1 - obj_mask_2d))).astype('uint8')

#     # ---- 标准化仅作用于第0通道 ----
#     if mean is not None and std is not None:
#         mean = np.asarray(mean)
#         std  = np.asarray(std)
#         std  = np.where(std == 0, 1.0, std)

#         # 支持 标量 或 (S,)
#         if mean.ndim == 0:
#             mean0 = mean.reshape(1, 1)
#         elif mean.shape == (S,):
#             mean0 = mean.reshape(1, S)
#         else:
#             raise ValueError(f"mean 需为标量或 (S,)，当前 {mean.shape}")

#         if std.ndim == 0:
#             std0 = std.reshape(1, 1)
#         elif std.shape == (S,):
#             std0 = std.reshape(1, S)
#         else:
#             raise ValueError(f"std 需为标量或 (S,)，当前 {std.shape}")

#         val0_std = (X_filled[..., 0] - mean0) / std0   # (T,S)
#     else:
#         val0_std = X_filled[..., 0]

#     # 第0通道：原始无观测处清零；时间通道：保持原值（不清零不隐藏）
#     val0_std = (val0_std * obj_mask_2d).astype(np.float32)

#     observed_data = X_filled.astype(np.float32).copy()   # (T,S,D)
#     observed_data[..., 0] = val0_std                     # 仅覆盖第0通道

#     # ---- 把 2D 掩码扩成 3D（按你的新规范）----
#     # 0通道：用 2D 掩码；其余通道：observed_mask=1（始终可见），gt_mask=0（不参与监督）
#     one  = np.ones((T, S), dtype=np.uint8)
#     zero = np.zeros((T, S), dtype=np.uint8)

#     observed_mask = np.zeros((T, S, D), dtype=np.uint8)
#     gt_mask       = np.zeros((T, S, D), dtype=np.uint8)
#     eval_mask     = np.zeros((T, S, D), dtype=np.uint8)
#     obj_mask      = np.zeros((T, S, D), dtype=np.uint8)

#     observed_mask[..., 0] = obj_mask_2d
#     gt_mask[..., 0]       = gt_mask_2d
#     eval_mask[..., 0]     = eval_mask_2d
#     obj_mask[..., 0]      = obj_mask_2d

#     if D > 1:
#         one_3d  = np.broadcast_to(one[..., None],  (T, S, D-1))   # (T,S,1) -> (T,S,D-1)
#         zero_3d = np.broadcast_to(zero[..., None], (T, S, D-1))
#         observed_mask[..., 1:] = one_3d
#         gt_mask[..., 1:]       = zero_3d
#         eval_mask[..., 1:]     = zero_3d
#         obj_mask[..., 1:]      = one_3d

#     # ---- 切分 ----
#     train_end = int(args.train_ratio * T)
#     val_end   = int((args.train_ratio + args.val_ratio) * T)

#     splits = {
#         "train": slice(0, train_end),
#         "val":   slice(train_end, val_end),
#         "test":  slice(val_end, T),
#     }

#     out = {}
#     for name, sl in splits.items():
#         out[name] = {
#             "observed_data": observed_data[sl],                # (T',S,D)
#             "observed_mask": observed_mask[sl],                # (T',S,D)
#             "gt_mask":       gt_mask[sl],                      # (T',S,D)
#         }

#     out["meta"] = {
#         "eval_mask": eval_mask,                                # (T,S,D)
#         "obj_mask":  obj_mask,                                 # (T,S,D)
#         "mean": None if mean is None else np.asarray(mean),
#         "std":  None if std  is None else np.asarray(std),
#     }
#     return out
