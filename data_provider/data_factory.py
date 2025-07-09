from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.data_loader_folsom import SlidingWindowLogicalDailySolarDataset, FastDataset
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    # 'Folsom': FastDataset,
    'Folsom': SlidingWindowLogicalDailySolarDataset,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False

        if args.data == 'Folsom':
            Data = data_dict[args.data]
            if flag == 'test':
                shuffle_flag = False
                drop_last = True

            elif flag == 'valid':
                shuffle_flag = True
                drop_last = True

            else:
                shuffle_flag = True
                drop_last = True

            # 计算 label_len_hours 以匹配模型期望
            # 对于5分钟频率数据，每小时12个点
            points_per_hour = 12
            label_len_hours = args.label_len // points_per_hour
            
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                seq_len_hours=args.seq_len_hours,
                pred_len_hours=args.pred_len_hours,
                label_len_hours=label_len_hours,  # 根据模型的label_len计算
                day_start_time_str="16:00",
                day_end_time_str="00:15",
                stride_minutes=10,
                features=args.features,
                target='ghi_5min',
                scale=True,
                timeenc=0,
                freq='5t'
            )
            print(len(data_set))
            data_loader = DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        else:
            data_set = Data(
                args = args,
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return data_set, data_loader

def data_provider_fast(args, flag):
    Data = data_dict[args.data]
    processed_root_path = '/opt/data/private/code/ICML25-TimeVLM-main/folsom'
    data_set = Data(processed_root_path, flag=flag)
    shuffle_flag = False
    drop_last = True
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
