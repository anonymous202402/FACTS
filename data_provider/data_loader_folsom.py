import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime as dt_datetime, time as dt_time, timedelta


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
                 seq_len_hours=18, pred_len_hours=6, label_len_hours=0,  
                 day_start_time_str="00:00",
                 day_end_time_str="23:55",
                 stride_minutes=None, 
                 features='S', target='ghi_5min', scale=True, timeenc=0, freq='5t'):

        self.seq_len_hours = seq_len_hours
        self.pred_len_hours = pred_len_hours
        self.label_len_hours = label_len_hours 
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
        self.label_len = self.label_len_hours * self.points_per_hour 
        self.total_window_len = self.seq_len + self.pred_len  

        if stride_minutes is None:
            self.stride_points = 1 
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

        self._processed_logical_days_data = [] 
        self._flat_index_map = [] 
        self.__read_and_prepare_data__()

    def __assign_logical_date_and_filter(self, df_full):
        df_full = df_full.sort_values('timestamp').reset_index(drop=True)
        logical_day_segments = []
        processed_indices = set()
        potential_start_dates = df_full['timestamp'].dt.normalize().unique()

        for calendar_date_obj in potential_start_dates:
            date_only_obj = pd.to_datetime(calendar_date_obj).date()
            current_logical_day_start_dt = dt_datetime.combine(date_only_obj, self.day_start_time)
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
        return logical_day_segments

    def __read_and_prepare_data__(self):
        self.scaler_img = StandardScaler()  
        self.scaler_ts = StandardScaler()  
        self.scaler_weather = StandardScaler()  
        try:
            df_raw_full = pd.read_csv(os.path.join(self.root_path, self.data_path))
        except FileNotFoundError:
            print(f"Error: DATA FILE {os.path.join(self.root_path, self.data_path)} NOT FOUND.")
            return
        df_raw_full['timestamp'] = pd.to_datetime(df_raw_full['timestamp'])
        all_logical_day_segments_dfs = self.__assign_logical_date_and_filter(df_raw_full)

        if not all_logical_day_segments_dfs:
            print("Warning: NO FULL LOGICAL DAY DATA SEGMENTS EXTRACTED FROM DATA.")
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
            print(f"Warning: flag '{self.flag}' NO LOGICAL DAY DATA SEGMENTS ASSIGNED.")
            return

        if self.scale:
            train_dfs_for_scaler = all_logical_day_segments_dfs[:train_end_idx]
            if not train_dfs_for_scaler:
                print("Warning: TRAINING DATA SEGMENT FOR SCALER IS EMPTY. WILL NOT SCALE.")
                self.scale = False
            else:
                train_values_img_list = [] 
                train_values_ts_list = []  
                train_values_weather_list = [] 
                cols_data_for_scale = []
                
                for seg_df_scaler in train_dfs_for_scaler:
                    if self.features == 'M' or self.features == 'MS':
                        if not cols_data_for_scale:
                            cols_data_for_scale = [col for col in seg_df_scaler.columns if
                                                   col not in ['timestamp', 'logical_date_key']]
                        all_values = seg_df_scaler[cols_data_for_scale].values
                        
                        img_values = all_values[:, :512]
                        ts_values = all_values[:, 512:554]
                        weather_values = all_values[:, 554:]
                        
                        train_values_img_list.append(img_values)
                        train_values_ts_list.append(ts_values)
                        train_values_weather_list.append(weather_values)
                    elif self.features == 'S':
                        if self.target not in seg_df_scaler.columns: continue
                        target_values = seg_df_scaler[[self.target]].values
                        train_values_ts_list.append(target_values)
                        
                if not train_values_img_list and not train_values_ts_list:
                    print("Warning: NO STANDARDIZED VALUES EXTRACTED FROM TRAINING SEGMENTS. WILL NOT SCALE.")
                    self.scale = False
                else:
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

        temp_cols_data_names = []  
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

            if np.isnan(day_values_orig).any() or np.isinf(day_values_orig).any():
                logical_date_key = segment_df['logical_date_key'].iloc[0]
                print(f"!!! Warning: IN ORIGINAL DATA ON LOGICAL DAY {logical_date_key} CONTAINS NaN/inf!")

            if self.scale:
                if self.features == 'M' or self.features == 'MS':
                    img_values = day_values_orig[:, :512]
                    ts_values = day_values_orig[:, 512:554]
                    weather_values = day_values_orig[:, 554:]
                    
                    img_values_processed = self.scaler_img.transform(img_values)
                    ts_values_processed = self.scaler_ts.transform(ts_values)
                    weather_values_processed = self.scaler_weather.transform(weather_values)
                    day_values_processed = np.hstack([img_values_processed, ts_values_processed, weather_values_processed])
                else:
                    day_values_processed = self.scaler_ts.transform(day_values_orig)
            else:
                day_values_processed = day_values_orig

            if np.isnan(day_values_processed).any() or np.isinf(day_values_processed).any():
                logical_date_key = segment_df['logical_date_key'].iloc[0]
                print(f"!!! Successfully located: ON LOGICAL DAY {logical_date_key}, STANDARDIZATION OPERATION PRODUCED NaN/inf!")

            segment_timestamps = segment_df['timestamp']
            day_stamp_values = None
            if self.timeenc == 0:
                stamps_list = [segment_timestamps.dt.month.values, segment_timestamps.dt.day.values,
                               segment_timestamps.dt.weekday.values, segment_timestamps.dt.hour.values]
                day_stamp_values = np.vstack(stamps_list).transpose()
            elif self.timeenc == 1:
                day_stamp_values = time_features(segment_timestamps, freq='h')  
            else:  # No time encoding
                day_stamp_values = np.empty((len(day_values_processed), 0))  

            current_day_tuple_idx = len(self._processed_logical_days_data)
            self._processed_logical_days_data.append(
                (day_values_processed, day_stamp_values, logical_date_key)
            )

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

        s_begin = start_in_day_idx
        s_end = s_begin + self.seq_len
        
        r_begin = s_end - self.label_len  
        r_end = r_begin + self.label_len + self.pred_len

        if r_end > len(day_specific_data):
            raise ValueError(
                f"INTERNAL ERROR: SLIDING WINDOW OUT OF BOUNDS. Day tuple idx {day_tuple_idx}, start_in_day_idx {start_in_day_idx}")



        x = day_specific_data[s_begin:s_end]     
        y = day_specific_data[r_begin:r_end]
        seq_x_img = x[:, :512]    
        seq_y_img = y[:, :512]    
        seq_x = x[:, 512:554]  
        seq_y = y[:, 512:554]
        seq_x_weather = x[:, 554:]
        seq_y_weather = y[:, 554:]
        seq_x_mark = day_specific_stamps[s_begin:s_end]
        seq_y_mark = day_specific_stamps[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_img, seq_y_img, seq_x_weather, seq_y_weather

    def __len__(self):
        return len(self._flat_index_map)

    def inverse_transform(self, data):
        if self.scale and hasattr(self, 'scaler_img') and hasattr(self, 'scaler_ts') and hasattr(self, 'scaler_weather'):
            if data.shape[-1] == 512:
                return self.scaler_img.inverse_transform(data)
            elif data.shape[-1] == 42:
                return self.scaler_ts.inverse_transform(data)
            elif data.shape[-1] == 554:
                img_data = data[..., :512]
                ts_data = data[..., 512:554]
                weather_data = data[..., 554:]
                img_inv = self.scaler_img.inverse_transform(img_data.reshape(-1, 512)).reshape(img_data.shape)
                ts_inv = self.scaler_ts.inverse_transform(ts_data.reshape(-1, 42)).reshape(ts_data.shape)
                weather_inv = self.scaler_weather.inverse_transform(weather_data.reshape(-1, 42)).reshape(weather_data.shape)
                return np.concatenate([img_inv, ts_inv, weather_inv], axis=-1)
        return data
