import os
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
import pandas as pd
from typing import Tuple, List, Optional


def sort_and_renumber_by_geo(
    df: pd.DataFrame,
    df_id: pd.DataFrame,
    geo_col: str = "geo_id",
    start_index: int = 0,
    strict: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    按 df_id[geo_col] 的出现顺序对 df 的列重排，并将列名重命名为
    start_index 起的连续整数（0,1,2,...）。返回 (df_sorted, id_map, df_only, geo_only)。

    关键修复点：无论 df.columns 是字符串还是整数，统一通过“int->原始列标签”的映射来安全选列。
    """

    # 1) 从 df_id 读取 geo 顺序并转为 int
    geo_series = pd.to_numeric(df_id[geo_col], errors="coerce").dropna().astype(int)
    geo_order: List[int] = geo_series.tolist()
    geo_set = set(geo_order)

    # 2) 构建 “int -> 原始列标签” 的稳健映射（允许 df.columns 为 str 或 int）
    #    - 先把列标签尝试转为数字；能转的映射到对应的原始标签
    col_numeric = pd.to_numeric(pd.Index(df.columns), errors="coerce")
    int_to_label = {}
    for lbl, num in zip(df.columns, col_numeric):
        if pd.notna(num):
            int_to_label[int(num)] = lbl  # 保留原始列标签（可能是字符串）

    df_int_ids = set(int_to_label.keys())

    # 3) 不匹配检查
    df_only = sorted(df_int_ids - geo_set)   # df 有但 geo 没有
    geo_only = sorted(geo_set - df_int_ids)  # geo 有但 df 没有

    if strict and (df_only or geo_only):
        raise ValueError(
            f"严格模式下 ID 不匹配：df_only={len(df_only)}, geo_only={len(geo_only)}；"
            f"样例 df_only={df_only[:5]}, geo_only={geo_only[:5]}"
        )

    # 4) 按 geo 顺序保留交集，并映射回“原始列标签”以进行选择
    keep_order_ints: List[int] = [gid for gid in geo_order if gid in df_int_ids]
    keep_labels: List = [int_to_label[gid] for gid in keep_order_ints]  # 原始列标签（str 或 int）

    # 5) 选择列并复制
    df_sorted = df.loc[:, keep_labels].copy()

    # 6) 重命名为 0..N-1（或从 start_index 开始）
    new_cols = list(range(start_index, start_index + len(keep_labels)))
    df_sorted.columns = new_cols

    # 7) 生成旧→新映射（按 geo 顺序的交集）
    id_map = pd.DataFrame({"old_id": keep_order_ints, "new_id": new_cols})

    return df_sorted, id_map, df_only, geo_only


def process_raw_dyna_data(df):
    """
    处理原始的单列dyna数据，并移除时区信息
    """
    print("原始数据形状:", df.shape)
    
    # 分割单列数据
    column_names = df.iloc[0, 0].split(',')
    print("检测到的列名:", column_names)
    
    # 从第二行开始分割数据
    split_data = []
    for i in range(1, len(df)):
        row = df.iloc[i, 0]
        if isinstance(row, str) and ',' in row:
            split_data.append(row.split(','))
    
    # 创建新的DataFrame
    if split_data:
        processed_df = pd.DataFrame(split_data, columns=column_names)
        
        # 数据清洗和类型转换
        # 首先转换为datetime，然后移除时区信息
        processed_df['time'] = pd.to_datetime(processed_df['time']).dt.tz_localize(None)
        
        processed_df['entity_id'] = processed_df['entity_id'].astype(str)
        processed_df['traffic_flow'] = pd.to_numeric(processed_df['traffic_flow'], errors='coerce')
        processed_df['dyna_id'] = pd.to_numeric(processed_df['dyna_id'], errors='coerce')
        
        return processed_df
    else:
        raise ValueError("无法分割数据")

def transform_dyna_to_timeseries(df):
    """
    将处理后的dyna数据转换为时间序列格式，确保时间格式为无时区
    """
    # 检查必要的列
    required_columns = ['time', 'entity_id', 'traffic_flow']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"缺少必要的列: {missing_columns}")
    
    print(f"转换前数据信息:")
    print(f"- 时间范围: {df['time'].min()} 到 {df['time'].max()}")
    print(f"- 时间格式: {type(df['time'].iloc[0])}")
    print(f"- 传感器数量: {df['entity_id'].nunique()}")
    print(f"- 总数据点数: {len(df)}")
    
    # 使用pivot进行转换
    try:
        timeseries_df = df.pivot(
            index='time',
            columns='entity_id',
            values='traffic_flow'
        )
    except Exception as e:
        print(f"pivot失败，使用pivot_table: {e}")
        # 如果有重复值，使用pivot_table
        timeseries_df = df.pivot_table(
            index='time',
            columns='entity_id',
            values='traffic_flow',
            aggfunc='first'
        )
    
    # 按时间排序
    timeseries_df = timeseries_df.sort_index()
    
    # 确保索引没有时区信息
    if timeseries_df.index.tz is not None:
        timeseries_df.index = timeseries_df.index.tz_localize(None)
    
    # 生成完整的时间序列（5分钟间隔）
    if not timeseries_df.empty:
        print(f"转换后原始形状: {timeseries_df.shape}")
        
        # 生成完整的时间索引（无时区）
        full_time_index = pd.date_range(
            start=timeseries_df.index.min(),
            end=timeseries_df.index.max(),
            freq='5min',
            tz=None  # 确保生成的时间索引无时区
        )
        
        # 重新索引以填充缺失的时间点
        timeseries_df = timeseries_df.reindex(full_time_index)
        
        print(f"填充时间序列后形状: {timeseries_df.shape}")
        print(f"时间点数量: {len(timeseries_df)}")
        print(f"传感器数量: {len(timeseries_df.columns)}")
        
        # 验证时间格式
        print(f"最终时间索引格式: {type(timeseries_df.index[0])}")
        print(f"时间样例: {timeseries_df.index[0]}")
    
    return timeseries_df

# 完整的处理流程
def complete_dyna_processing(original_df):
    """
    完整的dyna数据处理流程，确保输出无时区时间
    """
    print("=== 开始处理dyna数据 ===")
    
    # 步骤1: 处理原始单列数据
    processed_df = process_raw_dyna_data(original_df)
    print(f"处理后的数据形状: {processed_df.shape}")
    print("处理后的数据前5行:")
    print(processed_df.head())
    print(f"时间列类型: {type(processed_df['time'].iloc[0])}")
    
    # 步骤2: 转换为时间序列格式
    timeseries_df = transform_dyna_to_timeseries(processed_df)
    
    print("=== 处理完成 ===")
    print(f"最终时间序列数据形状: {timeseries_df.shape}")
    
    return timeseries_df

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
    return Path(__file__).resolve().parents[1]


DATA_ROOT = Path(os.getenv("IMPUTEST_DATA_ROOT", _repo_root()/ "datasets"))

def get_pems0X_data(args):

    if args.dataset_name == 'PeMS03':
        data_file = DATA_ROOT / f'{args.dataset_name}/PeMS03.h5'
    elif args.dataset_name == 'PeMS04':
        data_file = DATA_ROOT / f'{args.dataset_name}/PeMS04.h5'
    elif args.dataset_name == 'PeMS07':
        data_file = DATA_ROOT / f'{args.dataset_name}/PeMS07.h5'
    elif args.dataset_name == 'PeMS08':
        data_file = DATA_ROOT / f'{args.dataset_name}/PeMS08.h5'
    else:
        raise ValueError(f"Invalid dataset name for PeMS0X: {args.dataset_name}")
    
    df = pd.read_hdf(data_file)

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