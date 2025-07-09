import torch
import torch.nn as nn
import torch.nn.functional as F

class Gating_Unit(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, output_dim=1):
        """
        门控融合单元
        Args:
            feature_dim: 图像和时序特征的共同维度
            hidden_dim: 隐藏层维度
            output_dim: 输出预测维度
        """
        super(Gating_Unit, self).__init__()
        
        # 门控信号生成网络
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()  # 确保门控信号在[0,1]之间
        )
        
        # MLP预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, ts_features, img_features):
        """
        前向传播
        Args:
            ts_features: [batch_size, seq_len, feature_dim] 时序特征
            img_features: [batch_size, seq_len, feature_dim] 图像特征
        Returns:
            predictions: [batch_size, seq_len, output_dim] 预测结果
        """
        # 1. 生成门控信号 g = σ(W_gate * F_img + b_gate)
        gate_values = self.gate_network(img_features)  # [batch_size, seq_len, feature_dim]
        
        # 2. 门控融合: F_final = g⊙F_ts + (1-g)⊙F_img
        # 由于维度相同，直接使用img_features，无需变换
        fused_features = gate_values * ts_features + (1 - gate_values) * img_features
        
        # 3. 通过MLP预测头得到最终预测
        predictions = self.prediction_head(fused_features)
        
        return predictions

class Conditional_Unit(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, output_dim=1):
        """
        条件网络参数生成单元 (FiLM - Feature-wise Linear Modulation)
        Args:
            feature_dim: 图像和时序特征的共同维度
            hidden_dim: 隐藏层维度
            output_dim: 输出预测维度
        """
        super(Conditional_Unit, self).__init__()
        
        # 缩放因子生成网络 γ = MLP_γ(F_img)
        self.gamma_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()  # 确保γ在[0,1]范围内，避免过度放大
        )
        
        # 平移因子生成网络 β = MLP_β(F_img)
        self.beta_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()  # β可以为正负，允许双向调制
        )
        
        # MLP预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, ts_features, img_features):
        """
        前向传播
        Args:
            ts_features: [batch_size, seq_len, feature_dim] 时序特征
            img_features: [batch_size, seq_len, feature_dim] 图像特征
        Returns:
            predictions: [batch_size, seq_len, output_dim] 预测结果
        """
        # 1. 根据图像特征生成调制参数
        gamma = self.gamma_network(img_features)  # [batch_size, seq_len, feature_dim]
        beta = self.beta_network(img_features)    # [batch_size, seq_len, feature_dim]
        
        # 2. 对时序特征进行特征级线性调制: F_modulated = γ⊙F_ts + β
        modulated_features = gamma * ts_features + beta
        
        # 3. 通过MLP预测头得到最终预测
        predictions = self.prediction_head(modulated_features)
        
        return predictions

class Attentive_Unit(nn.Module):
    """交叉注意力融合单元"""
    def __init__(self, feature_dim, hidden_dim=128, output_dim=1, num_heads=8):
        super(Attentive_Unit, self).__init__()
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, ts_features, img_features):
        """
        前向传播，使用交叉注意力机制
        Args:
            ts_features: [batch_size, seq_len, feature_dim] 时序特征
            img_features: [batch_size, seq_len, feature_dim] 图像特征
        """
        # 1. 时序特征对图像特征的交叉注意力
        attended_ts, _ = self.cross_attention(
            query=ts_features,
            key=img_features,
            value=img_features
        )
        ts_enhanced = self.norm1(ts_features + attended_ts)
        
        # 2. 图像特征对时序特征的交叉注意力
        attended_img, _ = self.cross_attention(
            query=img_features,
            key=ts_enhanced,
            value=ts_enhanced
        )
        img_enhanced = self.norm2(img_features + attended_img)
        
        # 3. 融合特征的自注意力
        fused_features = (ts_enhanced + img_enhanced) / 2
        self_attended, _ = self.self_attention(
            query=fused_features,
            key=fused_features,
            value=fused_features
        )
        final_features = self.norm3(fused_features + self_attended)
        
        # 4. 前馈网络
        enhanced_features = final_features + self.ffn(final_features)
        
        # 5. 预测
        predictions = self.prediction_head(enhanced_features)
        
        return predictions

class Attentive_Unit_Img(nn.Module):
    """交叉注意力融合单元"""
    def __init__(self, feature_dim, hidden_dim=128, output_dim=1, num_heads=8):
        super(Attentive_Unit_Img, self).__init__()
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, ts_features, img_features_h, img_features_f):
        """
        前向传播，使用交叉注意力机制
        Args:
            ts_features: [batch_size, seq_len, feature_dim] 时序特征 (作为 Query)
            img_features_h: [batch_size, hist_seq_len, feature_dim] 历史图像特征
            img_features_f: [batch_size, future_seq_len, feature_dim] 未来图像（特权）特征
        """
        # 1. 准备交叉注意力的 Key 和 Value
        # 将历史图像特征和未来图像特征在序列维度上拼接，形成完整的视觉信息源
        img_features_kv = torch.cat([img_features_h, img_features_f], dim=1)

        # 2. 执行交叉注意力
        # ts_features 作为 Query，向 img_features_kv "查询" 相关信息
        # attn_output 是融合了图像信息后的新特征序列
        attn_output, _ = self.cross_attention(
            query=ts_features, 
            key=img_features_kv, 
            value=img_features_kv
        )
        
        # 3. 第一个残差连接和层归一化 (Add & Norm)
        # 将查询前的原始时序特征与查询结果相加，保留原始信息
        x = self.norm1(ts_features + attn_output)
        
        # 4. 执行自注意力
        # 让融合了视觉信息的新序列 x 内部的各个时间步之间进行信息交互和提炼
        self_attn_output, _ = self.self_attention(
            query=x, 
            key=x, 
            value=x
        )
        
        # 5. 第二个残差连接和层归一化
        x = self.norm2(x + self_attn_output)
        
        # 6. 通过前馈网络进行非线性变换
        ffn_output = self.ffn(x)
        
        # 7. 第三个残差连接和层归一化
        x = self.norm3(x + ffn_output)
        
        # 8. 对整个序列进行预测，保持时间维度
        # 不再只取最后一个时间步，而是对整个序列进行预测
        predictions = self.prediction_head(x)  # [batch_size, seq_len, output_dim]
        
        return predictions