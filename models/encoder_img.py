import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from math import ceil


class DynamicPatchEmbedding(nn.Module):
    def __init__(self, d_model, seg_len, stride, dropout):
        super(DynamicPatchEmbedding, self).__init__()
        self.d_model = d_model
        self.seg_len = seg_len
        self.stride = stride
        
        self.value_embedding = nn.Linear(seg_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_len):
        batch_size, n_vars, actual_len = x.shape
        
        pad_len = ceil(1.0 * seq_len / self.seg_len) * self.seg_len
        padding_len = pad_len - seq_len
        
        if padding_len > 0:
            x = F.pad(x, (0, padding_len), mode='replicate')
        
        x = x.unfold(dimension=-1, size=self.seg_len, step=self.stride)
        seg_num = x.shape[2]
        
        x = torch.reshape(x, (batch_size * n_vars, seg_num, self.seg_len))
        
        x = self.value_embedding(x)
        
        return self.dropout(x), n_vars, seg_num


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.dec_seg_num = self.pad_out_len // self.seg_len

        self.max_seg_num = 200 
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.max_seg_num, configs.d_model)
        )
        
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.dec_seg_num, configs.d_model)
        )
        
        self.enc_value_embedding = DynamicPatchEmbedding(
            configs.d_model, self.seg_len, self.seg_len, configs.dropout
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, self.dec_seg_num, configs.factor, configs.d_model, configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len, 
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                )
                for l in range(configs.e_layers + 1)
            ],
        )


        self.configs = configs

    def _create_encoder(self, in_seg_num):
        return Encoder(
            [
                scale_block(self.configs, 1 if l == 0 else self.win_size, self.configs.d_model, self.configs.n_heads, self.configs.d_ff,
                            1, self.configs.dropout,
                            in_seg_num if l == 0 else ceil(in_seg_num / self.win_size ** l), self.configs.factor
                            ) for l in range(self.configs.e_layers)
            ]
        )


    def forecast(self, x_enc, seq_len=None, return_intermediates=False):
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

 

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, return_intermediates=False, seq_len=None):
        if seq_len is None:
            seq_len = x_enc.shape[1]
            
        if return_intermediates:
            dec_out, intermediates = self.forecast(x_enc, seq_len, return_intermediates=True)
            final_output = dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
            return final_output, intermediates
        else:
            dec_out = self.forecast(x_enc, seq_len)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
            