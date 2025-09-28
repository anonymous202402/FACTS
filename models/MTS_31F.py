import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Model as encoder
from models.encoder_img import Model as encoder_img


def cal_similarity(x, y, delay=3):
    if x.shape != y.shape:
        raise ValueError("Input tensors x and y must have the same shape.")

    _, seq_len, _ = x.shape
    
    # Store cross-correlation scores for each delay
    correlations_per_delay = []

    # Iterate through all possible time delays p from 0 to delay
    for p in range(delay + 1):
        # Simulate time shift X(t+p) through slicing operation
        # This is equivalent to shifting x left by p units
        shifted_x = x[:, p:, :]
        
        # Slice y accordingly to match the length of shifted_x
        aligned_y = y[:, :seq_len - p, :]
        
        # Calculate element-wise product
        product = shifted_x * aligned_y
        
        # Sum along the sequence length dimension to get cross-correlation score at delay p
        # This is a discrete implementation of the integral formula in the paper
        # Output shape: (batch_size, feature_dim)
        correlation_at_p = torch.sum(product, dim=1)
        
        correlations_per_delay.append(correlation_at_p)

    # Stack all delay score tensors
    # New tensor shape: (delay + 1, batch_size, feature_dim)
    stacked_correlations = torch.stack(correlations_per_delay, dim=0)

    # Take maximum along delay dimension (dim=0) to find best matching score for each feature
    # torch.max returns (values, indices), we only need values
    max_correlation, _ = torch.max(stacked_correlations, dim=0)
    
    # Add sequence dimension to match (batch_size, seq_len, feature_dim) shape
    return max_correlation.unsqueeze(1)  # (batch_size, 1, feature_dim)

class Model(nn.Module):
    def __init__(self, args, args_img, args_weather):
        super(Model, self).__init__()
        self.args = args
        self.args_img = args_img
        self.branch_ts = encoder(args)
        args_img.seq_len = args
        self.branch_img = encoder_img(args_img)
        self.branch_weather = encoder_img(args_weather)
    
        
        # Project image and weather features to time series feature dimension separately
        self.img_projection = nn.Linear(args_img.c_out, args.c_out)  # 512 -> 42
        self.weather_projection = nn.Linear(args_weather.c_out, args.c_out)  # 6 -> 42
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # [time_series, image, weather]
        
    def forward(self, x_ts, x_img_h, x_img_f, x_weather_h, x_weather_f, ab=None):
        # Process time series features
        x_ts_f_pred = self.branch_ts(x_ts)

        # Concatenate x_img_h and x_img_f
        x_img = torch.cat([x_img_h, x_img_f], dim=1)
        x_weather = torch.cat([x_weather_h, x_weather_f], dim=1)
        
        # Process image sequences of different lengths separately with complete parameters
        x_img_f_pred = self.branch_img(x_img)
        x_img_f_pred = self.img_projection(x_img_f_pred)
        x_weather_f_pred = self.branch_weather(x_weather)
        x_weather_f_pred = self.weather_projection(x_weather_f_pred)
    
        # Calculate similarity and expand to sequence length dimension
        sim_img_ts = cal_similarity(x_img_f_pred, x_ts_f_pred)  # (batch_size, 1, feature_dim)
        sim_weather_ts = cal_similarity(x_weather_f_pred, x_ts_f_pred)  # (batch_size, 1, feature_dim)
        
        x_fused = x_ts_f_pred + x_img_f_pred * sim_img_ts + x_weather_f_pred * sim_weather_ts
                
        # Feature processing after fusion
        feat = x_fused.clone()
        return x_fused, sim_img_ts, sim_weather_ts, feat