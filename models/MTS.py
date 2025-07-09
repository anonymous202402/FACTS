import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Crossformer import Model as Crossformer
from models.confuser import Attentive_Unit

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


class Model(nn.Module):
    def __init__(self, args, args_img):
        super(Model, self).__init__()
        self.args = args
        self.args_img = args_img
        self.branch_ts = Crossformer(args)

    def forward(self, x, x_img):
        x_ts = self.branch_ts(x)
        return x_ts