import argparse
import torch
import torch.backends
from exp.exp_long_term_forecasting_teacher import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import copy

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Teacher Network Training for Multi-Modal Time Series Forecasting')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='teacher_MTS_3', help='model id')
    parser.add_argument('--model', type=str, default='MTS_3', help='teacher model name, should be MTS_3')

    # data loader
    parser.add_argument('--data', type=str, default='Folsom', help='dataset type')
    parser.add_argument('--root_path', type=str, help='root path of the data file')
    parser.add_argument('--data_path', type=str, help='data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='ghi_5min', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='5t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # model define
    parser.add_argument('--enc_in', type=int, default=42, help='encoder input size (time series features)')
    parser.add_argument('--dec_in', type=int, default=42, help='decoder input size (time series features)')
    parser.add_argument('--c_out', type=int, default=42, help='output size (time series features)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    
    # Folsom dataset specific parameters
    parser.add_argument('--seq_len_hours', type=int, default=4, help='input sequence length in hours')
    parser.add_argument('--pred_len_hours', type=int, default=2, help='prediction sequence length in hours')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='teacher_multimodal', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # Teacher network specific parameters (for future knowledge distillation)
    parser.add_argument('--kd_type', type=str, default='response', 
                        help='knowledge distillation type: response, relation, attention, contrastive, causal')
    parser.add_argument('--kd_loss_weight', type=float, default=0.1, help='knowledge distillation loss weight')
    
    # Multi-modal fusion parameters
    parser.add_argument('--fusion_type', type=str, default='attention', 
                        help='fusion method: attention, gate, concat')
    parser.add_argument('--img_feature_dim', type=int, default=512, help='image feature dimension')
    parser.add_argument('--weather_feature_dim', type=int, default=6, help='weather feature dimension')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')


    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Loss function parameters
    parser.add_argument('--loss_type', type=str, default='MSE', help='loss type', choices=['MSE', 'Huber', 'log_cosh', 'Weighted_MSE_MAE'])
    parser.add_argument('--alpha_weight', type=float, default=1.0, help='loss weight')
    parser.add_argument('--fuse_strategy', type=str, default=None, help='wether to fusion ab', choices=['no_img', 'no_weather'])

    args = parser.parse_args()
    
    # Device configuration
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU: cuda:{}'.format(args.gpu))
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using CPU or MPS')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Arguments for Teacher Network Training:')
    print_args(args)

    # Configure multi-modal parameters
    # Image branch parameter configuration
    args_img = copy.deepcopy(args)
    args_img.seq_len = args.seq_len + args.pred_len
    args_img.enc_in = args.img_feature_dim  # 512-dimensional image features
    args_img.dec_in = args.img_feature_dim
    args_img.c_out = args.img_feature_dim
    args_img.task_name = 'long_term_forecast'
    
    # Weather branch parameter configuration
    args_weather = copy.deepcopy(args)
    args_weather.seq_len = args.seq_len + args.pred_len
    args_weather.enc_in = args.weather_feature_dim  # 6-dimensional weather features
    args_weather.dec_in = args.weather_feature_dim
    args_weather.c_out = args.weather_feature_dim
    args_weather.task_name = 'long_term_forecast'

    # Ensure main branch (time series) configuration is correct
    if args.features == 'MS':
        args.c_out = 1  # Univariate prediction
    elif args.features == 'M':
        args.c_out = 42  # Multivariate prediction
    elif args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1

    # Validate configuration
    print(f"\n=== Multi-Modal Configuration ===")
    print(f"Main branch (time series): enc_in={args.enc_in}, c_out={args.c_out}")
    print(f"Image branch: enc_in={args_img.enc_in}, c_out={args_img.c_out}")
    print(f"Weather branch: enc_in={args_weather.enc_in}, c_out={args_weather.c_out}")
    print(f"Task: {args.task_name}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Sequence Length: {args.seq_len} ({args.seq_len_hours} hours)")
    print(f"Prediction Length: {args.pred_len} ({args.pred_len_hours} hours)")
    print("=" * 50)

    # Teacher network training only supports long-term forecasting task
    if args.task_name != 'long_term_forecast':
        print("Warning: Teacher network training is designed for long_term_forecast task.")
        print("Setting task_name to 'long_term_forecast'")
        args.task_name = 'long_term_forecast'

    # Experiment setup
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # Create experiment instance
            exp = Exp(args, args_img, args_weather)
            
            # Generate experiment setting identifier
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_fusion{}_imgdim{}_weatherdim{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.embed,
                args.fusion_type,
                args.img_feature_dim,
                args.weather_feature_dim,
                args.des, 
                ii)

            print('\n' + '='*100)
            print('>>>>>>>>>> Starting Teacher Network Training: {} <<<<<<<<<<'.format(setting))
            print('='*100)
            
            # Train teacher network
            exp.train(setting)

            print('\n' + '='*100)
            print('>>>>>>>>>> Testing Teacher Network: {} <<<<<<<<<<'.format(setting))
            print('='*100)
            
            # Test teacher network
            exp.test(setting)
            
            if args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
                
            print(f"Teacher network training iteration {ii+1}/{args.itr} completed.\n")
            
    else:
        # Test mode
        exp = Exp(args, args_img, args_weather)
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_fusion{}_imgdim{}_weatherdim{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.embed,
            args.fusion_type,
            args.img_feature_dim,
            args.weather_feature_dim,
            args.des, 
            ii)

        print('\n' + '='*100)
        print('>>>>>>>>>> Testing Trained Teacher Network: {} <<<<<<<<<<'.format(setting))
        print('='*100)
        
        exp.test(setting, test=1)
        
        if args.gpu_type == 'cuda':
            torch.cuda.empty_cache()

    print("\n🎉 Teacher Network Training/Testing completed successfully!")
    print("📁 Model checkpoints saved in:", args.checkpoints)
    print("📊 Results saved in: ./results/")
