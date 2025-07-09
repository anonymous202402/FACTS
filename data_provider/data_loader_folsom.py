import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime as dt_datetime, time as dt_time, timedelta


# 示例 time_features 函数
def time_features(dates_pd_series, freq='h'):
    df = pd.DataFrame({'timestamp': dates_pd_series})
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour
    if 't' in freq:
        df['minute'] = df['timestamp'].dt.minute
        return df[['month', 'day', 'weekday', 'hour', 'minute']].values
    else:
        return df[['month', 'day', 'weekday', 'hour']].values


class SlidingWindowLogicalDailySolarDataset(Dataset):
    def __init__(self, root_path, data_path='Target_intra-hour.csv', flag='train',
                 seq_len_hours=18, pred_len_hours=6, label_len_hours=0,  # 添加label_len_hours参数
                 day_start_time_str="00:00",
                 day_end_time_str="23:55",
                 stride_minutes=None,  # 新增：滑动窗口步长（分钟）。如果为None，则等于数据间隔
                 features='S', target='ghi_5min', scale=True, timeenc=0, freq='5t'):

        self.seq_len_hours = seq_len_hours
        self.pred_len_hours = pred_len_hours
        self.label_len_hours = label_len_hours  # 添加label_len_hours存储
        self.day_start_time = dt_datetime.strptime(day_start_time_str, "%H:%M").time()
        self.day_end_time = dt_datetime.strptime(day_end_time_str, "%H:%M").time()
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path

        if 't' in freq:
            self.minutes_interval = int(freq[:-1])
            self.points_per_hour = 60 // self.minutes_interval
        elif freq == 'h':
            self.minutes_interval = 60
            self.points_per_hour = 1
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

        self.seq_len = self.seq_len_hours * self.points_per_hour
        self.pred_len = self.pred_len_hours * self.points_per_hour
        self.label_len = self.label_len_hours * self.points_per_hour  # 计算label_len的点数
        self.total_window_len = self.seq_len + self.pred_len  # 一个完整样本(输入+预测)所需的总点数

        if stride_minutes is None:
            self.stride_points = 1  # 默认步长为1个数据点
        else:
            if stride_minutes % self.minutes_interval != 0:
                raise ValueError(
                    f"stride_minutes ({stride_minutes}) must be a multiple of data interval ({self.minutes_interval} minutes).")
            self.stride_points = stride_minutes // self.minutes_interval
            if self.stride_points <= 0:
                raise ValueError("stride_points must be positive.")

        start_dt_calc = dt_datetime.combine(dt_datetime.min.date(), self.day_start_time)
        end_dt_calc = dt_datetime.combine(dt_datetime.min.date(), self.day_end_time)

        if self.day_end_time <= self.day_start_time:
            end_dt_calc += timedelta(days=1)
        duration_seconds = (end_dt_calc - start_dt_calc).total_seconds()
        self.points_per_logical_day = int(duration_seconds / (self.minutes_interval * 60)) + 1

        if self.points_per_logical_day <= 0:
            raise ValueError("Logical day duration results in zero or negative points.")
        if self.total_window_len > self.points_per_logical_day:
            raise ValueError(f"Total sample length (seq_len + pred_len = {self.total_window_len} points) "
                             f"exceeds points in a logical day ({self.points_per_logical_day} points). "
                             f"Adjust seq_len_hours, pred_len_hours, or day start/end times.")

        print(f"逻辑日配置: 开始 {day_start_time_str}, 结束 {day_end_time_str} (相对开始日历日)")
        print(f"每个逻辑日预期点数: {self.points_per_logical_day} (基于 {self.minutes_interval} 分钟间隔)")
        print(f"输入序列点数 (seq_len): {self.seq_len}")
        print(f"预测序列点数 (pred_len): {self.pred_len}")
        print(f"滑动窗口步长 (stride_points): {self.stride_points}")

        self._processed_logical_days_data = []  # 存储每个逻辑日的 (data_array, stamp_array, logical_date_key)
        self._flat_index_map = []  # 存储 (day_idx, start_in_day_idx)
        self.__read_and_prepare_data__()

    def __assign_logical_date_and_filter(self, df_full):
        # (此函数与上一个版本中的 PreciseLogicalDailySolarDataset 基本相同)
        df_full = df_full.sort_values('timestamp').reset_index(drop=True)
        logical_day_segments = []
        processed_indices = set()
        potential_start_dates = df_full['timestamp'].dt.normalize().unique()

        for calendar_date_obj in potential_start_dates:
            date_only_obj = pd.to_datetime(calendar_date_obj).date()
            current_logical_day_start_dt = dt_datetime.combine(date_only_obj, self.day_start_time)
            # current_logical_day_start_dt = dt_datetime.combine(calendar_date_obj, self.day_start_time)
            current_logical_day_end_dt_calendar_date = calendar_date_obj
            if self.day_end_time <= self.day_start_time:
                current_logical_day_end_dt_calendar_date += np.timedelta64(1, 'D')

            current_logical_day_end_dt = dt_datetime.combine(pd.to_datetime(current_logical_day_end_dt_calendar_date).date(),
                                                             self.day_end_time)


            segment_mask = (df_full['timestamp'] >= current_logical_day_start_dt) & \
                           (df_full['timestamp'] <= current_logical_day_end_dt)
            segment_df = df_full[segment_mask]

            if not segment_df.empty:
                if segment_df.index[0] in processed_indices:
                    continue
                if len(segment_df) == self.points_per_logical_day:
                    segment_df = segment_df.copy()
                    segment_df['logical_date_key'] = pd.to_datetime(calendar_date_obj).date()
                    logical_day_segments.append(segment_df)
                    processed_indices.update(segment_df.index)
                # else:
                #     print(f"  逻辑日始于 {current_logical_day_start_dt} 数据段点数({len(segment_df)})与预期({self.points_per_logical_day})不符，跳过。")
        return logical_day_segments

    def __read_and_prepare_data__(self):
        self.scaler_img = StandardScaler()  # 图像变量标准化器
        self.scaler_ts = StandardScaler()   # 时序变量标准化器
        self.scaler_weather = StandardScaler()   # 天气变量标准化器
        try:
            df_raw_full = pd.read_csv(os.path.join(self.root_path, self.data_path))
        except FileNotFoundError:
            print(f"错误: 数据文件 {os.path.join(self.root_path, self.data_path)} 未找到。")
            return
        df_raw_full['timestamp'] = pd.to_datetime(df_raw_full['timestamp'])
        all_logical_day_segments_dfs = self.__assign_logical_date_and_filter(df_raw_full)

        if not all_logical_day_segments_dfs:
            print("警告: 未能从数据中提取任何完整的逻辑日数据段。")
            return

        num_total_segments = len(all_logical_day_segments_dfs)
        train_end_idx = max(0, int(num_total_segments * 0.7))
        val_end_idx = train_end_idx + max(0, int(num_total_segments * 0.15))

        if self.flag == 'train':
            current_segment_dfs = all_logical_day_segments_dfs[:train_end_idx]
        elif self.flag == 'val':
            current_segment_dfs = all_logical_day_segments_dfs[train_end_idx:val_end_idx]
        elif self.flag == 'test':
            current_segment_dfs = all_logical_day_segments_dfs[val_end_idx:]
        else:
            raise ValueError("Flag must be 'train', 'val', or 'test'.")

        if not current_segment_dfs:
            print(f"警告: flag '{self.flag}' 没有分配到任何逻辑日数据段。")
            return

        # --- 标准化 ---
        if self.scale:
            # 使用 all_logical_day_segments_dfs 的训练部分进行拟合
            train_dfs_for_scaler = all_logical_day_segments_dfs[:train_end_idx]
            if not train_dfs_for_scaler:
                print("警告: 用于标准化器的训练数据段为空。将不进行缩放。")
                self.scale = False
            else:
                train_values_img_list = []  # 图像变量训练数据
                train_values_ts_list = []   # 时序变量训练数据
                train_values_weather_list = []   # 天气变量训练数据
                cols_data_for_scale = []
                
                for seg_df_scaler in train_dfs_for_scaler:
                    if self.features == 'M' or self.features == 'MS':
                        if not cols_data_for_scale:
                            cols_data_for_scale = [col for col in seg_df_scaler.columns if
                                                   col not in ['timestamp', 'logical_date_key']]
                        all_values = seg_df_scaler[cols_data_for_scale].values
                        
                        # 分离图像变量（前512个）和时序变量（后512:554个）
                        # 天气变量（后554:596个）
                        img_values = all_values[:, :512]
                        ts_values = all_values[:, 512:554]
                        weather_values = all_values[:, 554:]
                        
                        train_values_img_list.append(img_values)
                        train_values_ts_list.append(ts_values)
                        train_values_weather_list.append(weather_values)
                    elif self.features == 'S':
                        if self.target not in seg_df_scaler.columns: continue
                        # 单变量情况下，根据target决定是图像还是时序
                        target_values = seg_df_scaler[[self.target]].values
                        train_values_ts_list.append(target_values)
                        
                if not train_values_img_list and not train_values_ts_list:
                    print("警告: 从训练段提取的标准化值为空。不缩放。")
                    self.scale = False
                else:
                    # 分别拟合两个标准化器
                    if train_values_img_list:
                        all_train_img_values = np.vstack(train_values_img_list)
                        if all_train_img_values.size > 0:
                            self.scaler_img.fit(all_train_img_values)
                            
                    if train_values_ts_list:
                        all_train_ts_values = np.vstack(train_values_ts_list)
                        if all_train_ts_values.size > 0:
                            self.scaler_ts.fit(all_train_ts_values)
                    if train_values_weather_list:
                        all_train_weather_values = np.vstack(train_values_weather_list)
                        if all_train_weather_values.size > 0:
                            self.scaler_weather.fit(all_train_weather_values)

        # --- 处理每个逻辑日的数据段并构建滑动窗口索引 ---
        temp_cols_data_names = []  # 用于M特征
        if current_segment_dfs:
            first_seg_df_for_cols = current_segment_dfs[0]
            if self.features == 'M' or self.features == 'MS':
                temp_cols_data_names = [col for col in first_seg_df_for_cols.columns if
                                        col not in ['timestamp', 'logical_date_key']]

        for day_idx, segment_df in enumerate(current_segment_dfs):
            logical_date_key = segment_df['logical_date_key'].iloc[0]

            day_values_orig = None
            if self.features == 'M' or self.features == 'MS':
                if not all(col in segment_df.columns for col in temp_cols_data_names):
                    continue
                day_values_orig = segment_df[temp_cols_data_names].values
            elif self.features == 'S':
                if self.target not in segment_df.columns:
                    continue
                day_values_orig = segment_df[[self.target]].values

            if day_values_orig is None: continue

            # 检查原始数据
            if np.isnan(day_values_orig).any() or np.isinf(day_values_orig).any():
                logical_date_key = segment_df['logical_date_key'].iloc[0]
                print(f"!!! 警告: 在逻辑日 {logical_date_key}，从CSV读取的原始数据就已包含 NaN/inf！")

            # 分别处理图像、时序变量、天气变量
            if self.scale:
                if self.features == 'M' or self.features == 'MS':
                    # 分离并分别标准化
                    img_values = day_values_orig[:, :512]
                    ts_values = day_values_orig[:, 512:554]
                    weather_values = day_values_orig[:, 554:]
                    
                    img_values_processed = self.scaler_img.transform(img_values)
                    ts_values_processed = self.scaler_ts.transform(ts_values)
                    weather_values_processed = self.scaler_weather.transform(weather_values)
                    # 重新组合
                    day_values_processed = np.hstack([img_values_processed, ts_values_processed, weather_values_processed])
                else:
                    # 单变量情况
                    day_values_processed = self.scaler_ts.transform(day_values_orig)
            else:
                day_values_processed = day_values_orig

            # 检查处理后的数据
            if np.isnan(day_values_processed).any() or np.isinf(day_values_processed).any():
                logical_date_key = segment_df['logical_date_key'].iloc[0]
                print(f"!!! 定位成功: 在逻辑日 {logical_date_key}，标准化操作产生了 NaN/inf！")

            # 时间戳特征 - 固定为4个特征以匹配模型期望
            segment_timestamps = segment_df['timestamp']
            day_stamp_values = None
            if self.timeenc == 0:
                # 始终使用4个基本时间特征，不包含分钟（避免维度不匹配）
                stamps_list = [segment_timestamps.dt.month.values, segment_timestamps.dt.day.values,
                               segment_timestamps.dt.weekday.values, segment_timestamps.dt.hour.values]
                day_stamp_values = np.vstack(stamps_list).transpose()
            elif self.timeenc == 1:
                # 修改freq参数，强制返回4个特征
                day_stamp_values = time_features(segment_timestamps, freq='h')  # 使用'h'避免分钟特征
            else:  # No time encoding
                day_stamp_values = np.empty((len(day_values_processed), 0))  # 保持维度一致性

            # 将处理好的日数据存储起来
            current_day_tuple_idx = len(self._processed_logical_days_data)
            self._processed_logical_days_data.append(
                (day_values_processed, day_stamp_values, logical_date_key)
            )

            # 在这个逻辑日内生成滑动窗口的起始索引
            # 一个逻辑日内能产生多少个样本 = (日总点数 - 完整样本总点数) // 步长 + 1
            num_samples_in_day = (self.points_per_logical_day - self.total_window_len) // self.stride_points + 1
            if num_samples_in_day > 0:
                for i in range(num_samples_in_day):
                    start_in_day_idx = i * self.stride_points
                    self._flat_index_map.append((current_day_tuple_idx, start_in_day_idx))

    def __getitem__(self, flat_index):
        if flat_index < 0 or flat_index >= len(self._flat_index_map):
            raise IndexError(f"Flat index {flat_index} out of bounds for dataset of size {len(self._flat_index_map)}")

        day_tuple_idx, start_in_day_idx = self._flat_index_map[flat_index]

        day_specific_data, day_specific_stamps, _ = self._processed_logical_days_data[day_tuple_idx]

        # 按照标准数据加载器的逻辑构建序列
        s_begin = start_in_day_idx
        s_end = s_begin + self.seq_len
        
        # 重要：需要包含label_len的重叠部分，以匹配Autoformer等模型的期望
        r_begin = s_end - self.label_len  # 包含重叠的label_len部分
        r_end = r_begin + self.label_len + self.pred_len  # 总长度为label_len+pred_len

        # 检查边界
        if r_end > len(day_specific_data):
            raise ValueError(
                f"内部错误: 滑动窗口越界。Day tuple idx {day_tuple_idx}, start_in_day_idx {start_in_day_idx}")



        x = day_specific_data[s_begin:s_end]     
        y = day_specific_data[r_begin:r_end]
        seq_x_img = x[:, :512]    
        seq_y_img = y[:, :512]    
        seq_x = x[:, 512:554]  
        seq_y = y[:, 512:554]
        seq_x_weather = x[:, 554:]
        seq_y_weather = y[:, 554:]
        seq_x_mark = day_specific_stamps[s_begin:s_end] # 长度: seq_len  
        seq_y_mark = day_specific_stamps[r_begin:r_end] # 长度: label_len + pred_len

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_img, seq_y_img, seq_x_weather, seq_y_weather

    def __len__(self):
        return len(self._flat_index_map)

    def inverse_transform(self, data):
        if self.scale and hasattr(self, 'scaler_img') and hasattr(self, 'scaler_ts') and hasattr(self, 'scaler_weather'):
            if data.shape[-1] == 512:  # 图像数据
                return self.scaler_img.inverse_transform(data)
            elif data.shape[-1] == 42:  # 时序数据
                return self.scaler_ts.inverse_transform(data)
            elif data.shape[-1] == 554:  # 完整数据
                img_data = data[..., :512]
                ts_data = data[..., 512:554]
                weather_data = data[..., 554:]
                img_inv = self.scaler_img.inverse_transform(img_data.reshape(-1, 512)).reshape(img_data.shape)
                ts_inv = self.scaler_ts.inverse_transform(ts_data.reshape(-1, 42)).reshape(ts_data.shape)
                weather_inv = self.scaler_weather.inverse_transform(weather_data.reshape(-1, 42)).reshape(weather_data.shape)
                return np.concatenate([img_inv, ts_inv, weather_inv], axis=-1)
        return data

import glob
import torch
class FastDataset(Dataset):
    def __init__(self, processed_root_path, flag='train'):
        self.processed_path = os.path.join(processed_root_path, flag)
        # 获取所有预处理好的.pt文件的路径列表
        self.file_paths = sorted(glob.glob(os.path.join(self.processed_path, "*.pt")))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # 唯一的任务：加载一个预处理好的文件，飞快！
        file_path = self.file_paths[index]
        data = torch.load(file_path)

        return data['seq_x'], data['seq_y'], data['seq_x_mark'], data['seq_y_mark']
