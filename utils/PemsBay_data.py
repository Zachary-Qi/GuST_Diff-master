import os
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


def build_temporal_features_df(
    df: pd.DataFrame,
    steps_per_day: int = 288,
    add_time_of_day: bool = True,
    add_day_of_week: bool = True,
    use_sin_cos: bool = False,
    per_sensor: bool = False,  # 保留但忽略：输出改为 3D ndarray
) -> np.ndarray:
    """
    将 (T,N) 的 DataFrame 转为 (T,N,C) 的 ndarray，并在最后一维拼接时间特征。

    默认输出 (T, N, 3)：
      [:,:,0] = 原始值（float32）
      [:,:,1] = time_of_day ∈ [0,1)
      [:,:,2] = day_of_week ∈ [0,1)   （周一=0/7, …, 周日=6/7）

    当 use_sin_cos=True 时，时间特征改为周期编码，输出 (T, N, 5)：
      [:,:,0] = 原始值
      [:,:,1] = tod_sin, [:,:,2] = tod_cos
      [:,:,3] = dow_sin, [:,:,4] = dow_cos

    参数
    ----
    df : (T, N) DataFrame，索引需为 DatetimeIndex（建议已对齐 5 分钟：df.asfreq("5T")）
    steps_per_day : 每天步数（5 分钟粒度=288）
    add_time_of_day / add_day_of_week : 是否添加相应时间特征（默认为都添加）
    use_sin_cos : 是否用 sin/cos 周期编码（为 True 时通道数变为 5）
    per_sensor : 仅保留以兼容旧签名；本实现总是返回 ndarray，而不是 DataFrame

    返回
    ----
    np.ndarray 形状 (T, N, C)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index 需为 pandas.DatetimeIndex。")

    T, N = df.shape
    values = df.to_numpy(dtype=np.float32)  # (T, N)

    # 计算时间特征（按真实时间）
    minutes = df.index.hour * 60 + df.index.minute
    # 一天中的步号 0..steps_per_day-1
    step_in_day = (minutes // (1440 // steps_per_day)).to_numpy()
    tod = (step_in_day / float(steps_per_day)).astype(np.float32)  # [0,1)

    # 一周中的星期 0..6（周一=0）→ 归一化 [0,1)
    dow_raw = df.index.dayofweek.to_numpy()
    dow = (dow_raw / 7.0).astype(np.float32)

    # 将 1D 时间特征平铺到 (T,N)
    tod_2d = np.tile(tod[:, None], (1, N)) if add_time_of_day else None
    dow_2d = np.tile(dow[:, None], (1, N)) if add_day_of_week else None

    # 组装通道
    ch_list = [values]  # 第 0 通道：原值

    if add_time_of_day:
        if use_sin_cos:
            tod_sin = np.sin(2 * np.pi * tod)[:, None]
            tod_cos = np.cos(2 * np.pi * tod)[:, None]
            ch_list.append(np.tile(tod_sin, (1, N)))
            ch_list.append(np.tile(tod_cos, (1, N)))
        else:
            ch_list.append(tod_2d)

    if add_day_of_week:
        if use_sin_cos:
            dow_sin = np.sin(2 * np.pi * dow)[:, None]
            dow_cos = np.cos(2 * np.pi * dow)[:, None]
            ch_list.append(np.tile(dow_sin, (1, N)))
            ch_list.append(np.tile(dow_cos, (1, N)))
        else:
            ch_list.append(dow_2d)

    # 拼到最后一维 -> (T, N, C)
    out = np.stack(ch_list, axis=-1).astype(np.float32)
    return out


def get_mean_std(df, train_ratio):
    data_len = len(df)
    train_data = df[:int(data_len*train_ratio)].values
    mean = np.mean(train_data, 0)
    std = np.std(train_data, 0)

    return mean, std
def _repo_root():
    # 当前文件在 .../ImputeST_v1/lib/data_utils/MetrLA_data.py
    # parents[2] => .../ImputeST_v1
    return Path(__file__).resolve().parents[1]


DATA_ROOT = Path(os.getenv("IMPUTEST_DATA_ROOT", _repo_root() / "datasets/pems_bay"))

def get_pemsbay_data(args):

    # 读取元数据
    df = pd.read_hdf(DATA_ROOT / 'pems_bay.h5')
    
    # print(f"{args.dataset_name} 数据形状: {df.shape}, 时间范围: {df.index[0]} ~ {df.index[-1]}")
    # print(f"{args.dataset_name} 样例数据: {df.head()}")
    
    mean, std = get_mean_std(df, args.train_ratio)

    # 把索引（时间戳）排序，得到从早到晚的列表
    datetime_idx = sorted(df.index)
    # 生成一个完整且等间隔 5 分钟的时间轴
    date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5min')
    # 把原表对齐到这个“理想时间轴”。
    df = df.reindex(index=date_range)
    
    feats = build_temporal_features_df(df, steps_per_day=288,      # 5min 粒度就是 288
                                    add_time_of_day=True, 
                                    add_day_of_week=True,
                                    use_sin_cos=False                      # 若想用周期 sin/cos，就改 True
                                    )
    # 返回邻接矩阵与数据
    return feats, mean, std