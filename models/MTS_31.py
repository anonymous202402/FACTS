from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Model as encoder
from models.encoder_img import Model as encoder_img
import random

def cal_similarity(x, y, delay=3):

    if x.shape != y.shape:
            raise ValueError("输入张量x和y的形状必须相同。")

    batch_size, seq_len, feature_dim = x.shape
    
    correlations_per_delay = []

    for p in range(delay + 1):
  
        shifted_x = x[:, p:, :]
        
        aligned_y = y[:, :seq_len - p, :]
        
        product = shifted_x * aligned_y
        
       
        correlation_at_p = torch.sum(product, dim=1)
        
        correlations_per_delay.append(correlation_at_p)

    
    stacked_correlations = torch.stack(correlations_per_delay, dim=0)

    
    max_correlation, _ = torch.max(stacked_correlations, dim=0)
    
    return max_correlation.unsqueeze(1)  # (batch_size, 1, feature_dim)

class Model(nn.Module):
    def __init__(self, args, args_img, args_weather):
        super(Model, self).__init__()
        self.args = args
        self.args_img = args_img
        self.branch_ts = encoder(args)
        self.branch_img = encoder_img(args_img)
        self.branch_weather = encoder_img(args_weather)
        
        self.img_projection = nn.Linear(args_img.c_out, args.c_out)  # 512 -> 42
        self.weather_projection = nn.Linear(args_weather.c_out, args.c_out)  # 6 -> 42
        self.tors = args.tors
        self.is_training = args.is_training
        self.modality_dropout_rate = args.modality_dropout_rate

  
        
    def forward(self, x_ts, x_img_h, x_img_f, x_weather_h, x_weather_f, used_data=None):
        if self.is_training:
            x_ts_f_pred = self.branch_ts(x_ts)

            x_img_f_pred = self.branch_img(x_img_h)
            x_img_f_pred = self.img_projection(x_img_f_pred)
            
            x_weather_f_pred = self.branch_weather(x_weather_h)
            x_weather_f_pred = self.weather_projection(x_weather_f_pred)

            sim_img_ts = cal_similarity(x_img_f_pred, x_ts_f_pred)  # (batch_size, 1, feature_dim)
            sim_weather_ts = cal_similarity(x_weather_f_pred, x_ts_f_pred)  # (batch_size, 1, feature_dim)
            
            
            if random.random() < self.modality_dropout_rate:
                x_img_f_pred = torch.zeros_like(x_img_f_pred)
                drop_img = torch.tensor(0.0, device=x_ts_f_pred.device)
            else:
                drop_img = torch.tensor(1.0, device=x_ts_f_pred.device)
            if random.random() < self.modality_dropout_rate:
                x_weather_f_pred = torch.zeros_like(x_weather_f_pred)
                drop_weather = torch.tensor(0.0, device=x_ts_f_pred.device)
            else:
                drop_weather = torch.tensor(1.0, device=x_ts_f_pred.device)
            x = x_ts_f_pred + x_img_f_pred * sim_img_ts + x_weather_f_pred * sim_weather_ts

            feat = x.clone()

            return x, sim_img_ts, sim_weather_ts, drop_img, drop_weather, feat
        
        else:
            x_ts_f_pred = self.branch_ts(x_ts)

            x_img_f_pred = self.branch_img(x_img_h)
            x_img_f_pred = self.img_projection(x_img_f_pred)  
            
            x_weather_f_pred = self.branch_weather(x_weather_h)
            x_weather_f_pred = self.weather_projection(x_weather_f_pred)
            
            sim_img_ts = cal_similarity(x_img_f_pred, x_ts_f_pred)  # (batch_size, 1, feature_dim)
            sim_weather_ts = cal_similarity(x_weather_f_pred, x_ts_f_pred)  # (batch_size, 1, feature_dim)
            
            
            
            if used_data == 'img':
                x = x_ts_f_pred + x_img_f_pred * sim_img_ts
            elif used_data == 'weather':
                x = x_ts_f_pred + x_weather_f_pred * sim_weather_ts
            elif used_data == 'ts':
                x = x_ts_f_pred
            elif used_data == 'both':
                x = x_ts_f_pred + x_img_f_pred * sim_img_ts + x_weather_f_pred * sim_weather_ts
            else:
                # Default behavior when used_data is None or not specified
                x = x_ts_f_pred + x_img_f_pred * sim_img_ts + x_weather_f_pred * sim_weather_ts
            
            # Return consistent format for evaluation mode
            feat = x.clone()
            return x, sim_img_ts, sim_weather_ts, feat