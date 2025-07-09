import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead
from math import ceil


class DynamicPatchEmbedding(nn.Module):
    """支持动态序列长度的PatchEmbedding"""
    def __init__(self, d_model, seg_len, stride, dropout):
        super(DynamicPatchEmbedding, self).__init__()
        self.d_model = d_model
        self.seg_len = seg_len
        self.stride = stride
        
        # 值嵌入层
        self.value_embedding = nn.Linear(seg_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_len):
        """
        x: [batch_size, n_vars, seq_len]
        seq_len: 实际序列长度
        """
        batch_size, n_vars, actual_len = x.shape
        
        # 动态计算填充长度
        pad_len = ceil(1.0 * seq_len / self.seg_len) * self.seg_len
        padding_len = pad_len - seq_len
        
        # 动态填充
        if padding_len > 0:
            x = F.pad(x, (0, padding_len), mode='replicate')
        
        # 进行分片
        x = x.unfold(dimension=-1, size=self.seg_len, step=self.stride)
        seg_num = x.shape[2]
        
        # 重塑为[batch_size * n_vars, seg_num, seg_len]
        x = torch.reshape(x, (batch_size * n_vars, seg_num, self.seg_len))
        
        # 值嵌入
        x = self.value_embedding(x)
        
        return self.dropout(x), n_vars, seg_num


class Model(nn.Module):
    """
    修改后的Crossformer_img，支持动态序列长度
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        # 计算预测任务的解码器参数（固定的）
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.dec_seg_num = self.pad_out_len // self.seg_len

        # 使用最大分段数来初始化编码器位置编码
        self.max_seg_num = 200  # 设置一个足够大的最大分段数
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.max_seg_num, configs.d_model)
        )
        
        # 解码器位置编码（固定基于预测长度）
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.dec_seg_num, configs.d_model)
        )
        
        # 动态嵌入层
        self.enc_value_embedding = DynamicPatchEmbedding(
            configs.d_model, self.seg_len, self.seg_len, configs.dropout
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # 创建固定的解码器（用于预测任务）
        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, self.dec_seg_num, configs.factor, configs.d_model, configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,  # 每个分段输出seg_len长度
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                )
                for l in range(configs.e_layers + 1)
            ],
        )

        # 为非预测任务创建头部网络的配置
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # 计算一个合理的默认分段数用于头部网络
            default_seq_len = getattr(configs, 'seq_len', 96)
            default_seg_num = ceil(default_seq_len / self.seg_len)
            out_seg_num = default_seg_num
            for l in range(configs.e_layers):
                if l > 0:
                    out_seg_num = ceil(out_seg_num / self.win_size)
            head_nf = configs.d_model * out_seg_num
            self.head = FlattenHead(configs.enc_in, head_nf, default_seq_len,
                                  head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            default_seq_len = getattr(configs, 'seq_len', 96)
            default_seg_num = ceil(default_seq_len / self.seg_len)
            out_seg_num = default_seg_num
            for l in range(configs.e_layers):
                if l > 0:
                    out_seg_num = ceil(out_seg_num / self.win_size)
            head_nf = configs.d_model * out_seg_num
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(head_nf * configs.enc_in, configs.num_class)

        # 配置参数
        self.configs = configs

    def _create_encoder(self, in_seg_num):
        """动态创建编码器"""
        return Encoder(
            [
                scale_block(self.configs, 1 if l == 0 else self.win_size, self.configs.d_model, self.configs.n_heads, self.configs.d_ff,
                            1, self.configs.dropout,
                            in_seg_num if l == 0 else ceil(in_seg_num / self.win_size ** l), self.configs.factor
                            ) for l in range(self.configs.e_layers)
            ]
        )

    def _create_head(self, in_seg_num):
        """动态创建任务头部"""
        # 计算编码器最终输出的分段数（用于头部网络）
        out_seg_num = in_seg_num
        for l in range(self.configs.e_layers):
            if l > 0:
                out_seg_num = ceil(out_seg_num / self.win_size)
        
        head_nf = self.configs.d_model * out_seg_num
        
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            return FlattenHead(self.configs.enc_in, head_nf, self.configs.seq_len,
                             head_dropout=self.configs.dropout)
        elif self.task_name == 'classification':
            return nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Dropout(self.configs.dropout),
                nn.Linear(head_nf * self.configs.enc_in, self.configs.num_class)
            )
        return None

    def forecast(self, x_enc, seq_len=None, return_intermediates=False):
        """动态预测方法"""
        if seq_len is None:
            seq_len = x_enc.shape[1]
        
        # 动态嵌入
        x_enc, n_vars, seg_num = self.enc_value_embedding(x_enc.permute(0, 2, 1), seq_len)
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        
        # 使用动态位置编码
        pos_enc = self.enc_pos_embedding[:, :, :seg_num, :]
        x_enc += pos_enc
        x_enc = self.pre_norm(x_enc)
        
        # 动态创建编码器
        encoder = self._create_encoder(seg_num)
        
        if torch.cuda.is_available() and x_enc.is_cuda:
            encoder = encoder.cuda()
            
        enc_out, attns = encoder(x_enc)

        # 使用固定的解码器和解码器位置编码
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        
        if return_intermediates:
            dec_out, dec_intermediates = self.decoder(dec_in, enc_out, return_intermediates=True)
            
            intermediates = {
                'encoder_features': enc_out,
                'decoder_features': dec_intermediates['decoder_features'],
                'layer_predictions': dec_intermediates['layer_predictions'],
            }
            return dec_out, intermediates
        else:
            dec_out = self.decoder(dec_in, enc_out)
            return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, seq_len=None):
        """动态插值方法"""
        if seq_len is None:
            seq_len = x_enc.shape[1]
            
        x_enc, n_vars, seg_num = self.enc_value_embedding(x_enc.permute(0, 2, 1), seq_len)
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        
        pos_enc = self.enc_pos_embedding[:, :, :seg_num, :]
        x_enc += pos_enc
        x_enc = self.pre_norm(x_enc)
        
        encoder = self._create_encoder(seg_num)
        
        if torch.cuda.is_available() and x_enc.is_cuda:
            encoder = encoder.cuda()
            
        enc_out, attns = encoder(x_enc)

        # 对于插值任务，使用头部网络
        if hasattr(self, 'head'):
            dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        else:
            # 如果没有预定义的头部网络，动态创建
            head = self._create_head(seg_num)
            if head is not None:
                if torch.cuda.is_available() and x_enc.is_cuda:
                    head = head.cuda()
                dec_out = head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
            else:
                # 如果无法创建头部网络，返回编码器最后一层的输出
                dec_out = enc_out[-1].mean(dim=2)  # [B, D]
        return dec_out

    def anomaly_detection(self, x_enc, seq_len=None):
        """动态异常检测方法"""
        if seq_len is None:
            seq_len = x_enc.shape[1]
            
        x_enc, n_vars, seg_num = self.enc_value_embedding(x_enc.permute(0, 2, 1), seq_len)
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        
        pos_enc = self.enc_pos_embedding[:, :, :seg_num, :]
        x_enc += pos_enc
        x_enc = self.pre_norm(x_enc)
        
        encoder = self._create_encoder(seg_num)
        
        if torch.cuda.is_available() and x_enc.is_cuda:
            encoder = encoder.cuda()
            
        enc_out, attns = encoder(x_enc)

        # 对于异常检测任务，使用头部网络
        if hasattr(self, 'head'):
            dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        else:
            # 如果没有预定义的头部网络，动态创建
            head = self._create_head(seg_num)
            if head is not None:
                if torch.cuda.is_available() and x_enc.is_cuda:
                    head = head.cuda()
                dec_out = head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
            else:
                # 如果无法创建头部网络，返回编码器最后一层的输出
                dec_out = enc_out[-1].mean(dim=2)  # [B, D]
        return dec_out

    def classification(self, x_enc, x_mark_enc, seq_len=None):
        """动态分类方法"""
        if seq_len is None:
            seq_len = x_enc.shape[1]
            
        x_enc, n_vars, seg_num = self.enc_value_embedding(x_enc.permute(0, 2, 1), seq_len)
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        
        pos_enc = self.enc_pos_embedding[:, :, :seg_num, :]
        x_enc += pos_enc
        x_enc = self.pre_norm(x_enc)
        
        encoder = self._create_encoder(seg_num)
        
        if torch.cuda.is_available() and x_enc.is_cuda:
            encoder = encoder.cuda()
            
        enc_out, attns = encoder(x_enc)
        
        # 对于分类任务，使用预定义的投影层
        if hasattr(self, 'flatten') and hasattr(self, 'projection'):
            output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
            output = self.dropout(output)
            output = output.reshape(output.shape[0], -1)
            output = self.projection(output)
        else:
            # 如果没有预定义的分类层，动态创建
            head = self._create_head(seg_num)
            if head is not None:
                if torch.cuda.is_available() and x_enc.is_cuda:
                    head = head.cuda()
                output = head(enc_out[-1].permute(0, 1, 3, 2))
            else:
                # 如果无法创建头部网络，返回编码器最后一层的平均输出
                output = enc_out[-1].mean(dim=2).mean(dim=2)  # [B, D] -> [B]
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, return_intermediates=False, seq_len=None):
        """
        前向传播方法，支持动态序列长度
        
        Args:
            x_enc: 输入序列 [batch_size, seq_len, features]
            seq_len: 可选，指定序列长度（如果None，则使用x_enc.shape[1]）
            其他参数保持不变
        """
        if seq_len is None:
            seq_len = x_enc.shape[1]
            
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if return_intermediates:
                dec_out, intermediates = self.forecast(x_enc, seq_len, return_intermediates=True)
                final_output = dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
                return final_output, intermediates
            else:
                dec_out = self.forecast(x_enc, seq_len)
                return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
                
        elif self.task_name == 'imputation':
            if return_intermediates:
                x_enc_emb, n_vars, seg_num = self.enc_value_embedding(x_enc.permute(0, 2, 1), seq_len)
                embedded_input = x_enc_emb.clone()
                x_enc = rearrange(x_enc_emb, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
                
                pos_enc = self.enc_pos_embedding[:, :, :seg_num, :]
                x_enc += pos_enc
                x_enc = self.pre_norm(x_enc)
                
                encoder = self._create_encoder(seg_num)
                
                if torch.cuda.is_available() and x_enc.is_cuda:
                    encoder = encoder.cuda()
                    
                enc_out, attns = encoder(x_enc)
                
                if hasattr(self, 'head'):
                    dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
                else:
                    head = self._create_head(seg_num)
                    if head is not None:
                        if torch.cuda.is_available() and x_enc.is_cuda:
                            head = head.cuda()
                        dec_out = head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
                    else:
                        # 如果无法创建头部网络，返回编码器最后一层的输出
                        dec_out = enc_out[-1].mean(dim=2)  # [B, D]
                
                intermediates = {
                    'embedded_input': embedded_input,
                    'encoder_features': enc_out,
                    'encoder_attentions': attns
                }
                return dec_out, intermediates
            else:
                dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, seq_len)
                return dec_out
                
        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, seq_len)
            return dec_out
            
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc, seq_len)
            return dec_out
            
        return None