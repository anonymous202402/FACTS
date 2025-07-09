import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Crossformer import Model as Crossformer
from models.confuser import Attentive_Unit
from models.Crossformer_img import Model as Crossformer_img
from models.confuser import Attentive_Unit_Img

class Model(nn.Module):
    def __init__(self, args, args_img):
        super(Model, self).__init__()
        self.args = args
        self.args_img = args_img
        self.branch_ts = Crossformer(args)
        self.branch_img = Crossformer_img(args_img)
        # 需要projection_layer将图像特征从512维投影到42维
        self.projection_layer = nn.Linear(args_img.c_out, args.c_out)
        self.confuser = Attentive_Unit_Img(feature_dim=self.args.c_out, hidden_dim=self.args.c_out, output_dim=self.args.c_out, num_heads=6)

    def forward(self, x_ts, x_img_h, x_img_f):
        """
        x_ts: [batch_size, seq_len, feature_dim] 时序特征 (作为 Query)
        x_img_h: [batch_size, hist_seq_len, feature_dim] 历史图像特征
        x_img_f: [batch_size, future_seq_len, feature_dim] 未来图像（特权）特征
        """
        # 处理时序特征
        x_ts = self.branch_ts(x_ts)
        
        # 分别处理不同长度的图像序列，传入完整参数
        hist_seq_len = x_img_h.shape[1]
        future_seq_len = x_img_f.shape[1]
        
        x_img_h = self.branch_img(x_img_h, seq_len=hist_seq_len)
        x_img_f = self.branch_img(x_img_f, seq_len=future_seq_len)
        
        # 投影图像特征到与时序特征相同的维度
        x_img_h = self.projection_layer(x_img_h)
        x_img_f = self.projection_layer(x_img_f)
        
        # 使用confuser进行融合
        x_ts = self.confuser(x_ts, x_img_h, x_img_f)
        return x_ts

# class Model(nn.Module):
#     def __init__(self, args, args_img):
#         super(Model, self).__init__()
#         self.args = args
#         self.args_img = args_img
#         self.branch_ts = Crossformer(args)
#         self.branch_img = Crossformer(args_img)
#         self.projection_layer = nn.Linear(args_img.c_out, args.c_out)
#         self.confuser = Attentive_Unit(feature_dim=self.args.c_out, hidden_dim=self.args.c_out, output_dim=self.args.c_out, num_heads=6)

#     def forward(self, x, x_img):
#         x_ts = self.branch_ts(x)
#         x_img = self.branch_img(x_img)
#         x_img = self.projection_layer(x_img)
#         x_ts = self.confuser(x_ts, x_img)
#         return x_ts