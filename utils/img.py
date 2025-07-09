import torch
import torch.nn as nn
import torch.nn.functional as F

class temporal_contrastive_loss(nn.Module):
    def __init__(self, temperature=0.1, positive_window=1, negative_window=10):
        super(temporal_contrastive_loss, self).__init__()
        self.temperature = temperature
        self.positive_window = positive_window  # 正样本时间窗口大小
        self.negative_window = negative_window  # 负样本时间窗口大小

    def forward(self, img_feats):
        """
        Args:
            img_feats: [batch_size, seq_len, feature_dim] 图像特征序列
        Returns:
            contrastive_loss: 时间对比损失
        """
        batch_size, seq_len, feature_dim = img_feats.shape
        
        # L2标准化特征向量
        img_feats_norm = F.normalize(img_feats, dim=-1)
        
        total_loss = 0.0
        valid_anchors = 0
        
        # 对每个时间点作为锚点进行对比学习
        for t in range(seq_len):
            # 1. 定义正样本：时间上相近的特征
            positive_indices = []
            for offset in range(-self.positive_window, self.positive_window + 1):
                pos_t = t + offset
                if 0 <= pos_t < seq_len and pos_t != t:  # 排除自己
                    positive_indices.append(pos_t)
            
            # 2. 定义负样本：时间上相距较远的特征
            negative_indices = []
            for offset in range(-seq_len, seq_len):
                neg_t = t + offset
                if (0 <= neg_t < seq_len and 
                    abs(offset) >= self.negative_window and 
                    neg_t != t):
                    negative_indices.append(neg_t)
            
            # 如果没有足够的正样本或负样本，跳过这个锚点
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
                
            # 3. 提取锚点特征
            anchor_feats = img_feats_norm[:, t, :]  # [batch_size, feature_dim]
            
            # 4. 提取正样本特征
            positive_feats = img_feats_norm[:, positive_indices, :]  # [batch_size, num_pos, feature_dim]
            
            # 5. 提取负样本特征
            negative_feats = img_feats_norm[:, negative_indices, :]  # [batch_size, num_neg, feature_dim]
            
            # 6. 计算相似度
            # 正样本相似度
            pos_sim = torch.einsum('bd,bpd->bp', anchor_feats, positive_feats)  # [batch_size, num_pos]
            
            # 负样本相似度
            neg_sim = torch.einsum('bd,bnd->bn', anchor_feats, negative_feats)  # [batch_size, num_neg]
            
            # 7. 应用温度系数
            pos_sim = pos_sim / self.temperature
            neg_sim = neg_sim / self.temperature
            
            # 8. 计算InfoNCE损失
            for b in range(batch_size):
                # 为当前batch样本计算损失
                pos_scores = pos_sim[b]  # [num_pos]
                neg_scores = neg_sim[b]  # [num_neg]
                
                # 对于每个正样本，计算与所有负样本的InfoNCE损失
                for pos_score in pos_scores:
                    # 计算分子（正样本得分）
                    numerator = torch.exp(pos_score)
                    
                    # 计算分母（正样本 + 所有负样本得分）
                    all_neg_scores = torch.exp(neg_scores)
                    denominator = numerator + torch.sum(all_neg_scores)
                    
                    # InfoNCE损失
                    loss = -torch.log(numerator / denominator)
                    total_loss += loss
                    valid_anchors += 1
        
        # 返回平均损失
        if valid_anchors > 0:
            return total_loss / valid_anchors
        else:
            return torch.tensor(0.0, device=img_feats.device, requires_grad=True)

class enhanced_temporal_contrastive_loss(nn.Module):
    """优化版本：使用向量化计算提高效率"""
    def __init__(self, temperature=0.1, positive_window=1, negative_window=10):
        super(enhanced_temporal_contrastive_loss, self).__init__()
        self.temperature = temperature
        self.positive_window = positive_window
        self.negative_window = negative_window

    def forward(self, img_feats):
        """
        Args:
            img_feats: [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len, feature_dim = img_feats.shape
        
        # L2标准化
        img_feats_norm = F.normalize(img_feats, dim=-1)
        
        # 计算所有时间点之间的相似度矩阵
        # similarity_matrix[i,j] = cos_sim(feat_i, feat_j)
        similarity_matrix = torch.einsum('bif,bjf->bij', img_feats_norm, img_feats_norm)
        similarity_matrix = similarity_matrix / self.temperature
        
        total_loss = 0.0
        valid_pairs = 0
        
        for t in range(seq_len):
            # 定义正样本mask
            pos_mask = torch.zeros(seq_len, dtype=torch.bool, device=img_feats.device)
            for offset in range(-self.positive_window, self.positive_window + 1):
                pos_t = t + offset
                if 0 <= pos_t < seq_len and pos_t != t:
                    pos_mask[pos_t] = True
            
            # 定义负样本mask
            neg_mask = torch.zeros(seq_len, dtype=torch.bool, device=img_feats.device)
            for offset in range(-seq_len, seq_len):
                neg_t = t + offset
                if (0 <= neg_t < seq_len and 
                    abs(offset) >= self.negative_window and 
                    neg_t != t):
                    neg_mask[neg_t] = True
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            # 对每个batch计算损失
            for b in range(batch_size):
                pos_scores = similarity_matrix[b, t, pos_mask]  # 正样本得分
                neg_scores = similarity_matrix[b, t, neg_mask]  # 负样本得分
                
                # 计算InfoNCE损失
                for pos_score in pos_scores:
                    numerator = torch.exp(pos_score)
                    denominator = numerator + torch.sum(torch.exp(neg_scores))
                    loss = -torch.log(numerator / denominator)
                    total_loss += loss
                    valid_pairs += 1
        
        return total_loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0, device=img_feats.device, requires_grad=True)