import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

def pdist(e):
    # 计算批次内样本间的欧氏距离
    pw_distances = torch.cdist(e, e, p=2.0)
    # 归一化，使其对尺度不敏感
    pw_distances = pw_distances / pw_distances.mean()
    return pw_distances

class response_kd(nn.Module):
    def __init__(self):
        super(response_kd, self).__init__()

    def forward(self, ts, img):
        return F.mse_loss(ts, img)

class relation_kd(nn.Module):
    def __init__(self):
        super(relation_kd, self).__init__()

    def forward(self, ts, img):
        dist_mat_ts = pdist(ts)
        dist_mat_img = pdist(img)
        return F.mse_loss(dist_mat_ts, dist_mat_img)

class attention_kd(nn.Module):
    def __init__(self):
        super(attention_kd, self).__init__()

    def forward(self, ts, img):
        return F.mse_loss(ts, img)

class contrastive_kd(nn.Module):
    def __init__(self, temperature=0.1, contrast_type='instance'):
        super(contrastive_kd, self).__init__()
        self.temperature = temperature
        self.contrast_type = contrast_type

    def forward(self, ts, img):
        # ts, img: [batch_size, seq_len, feature_dim]
        batch_size, seq_len, feature_dim = ts.shape

        if self.contrast_type == 'instance':
            # 方案1：使用reshape替代view（推荐）
            ts_flat = ts.reshape(-1, feature_dim)  # [batch_size * seq_len, feature_dim]
            img_flat = img.reshape(-1, feature_dim)  # [batch_size * seq_len, feature_dim]
            
            # 方案2：先确保连续性再使用view
            # ts_flat = ts.contiguous().view(-1, feature_dim)
            # img_flat = img.contiguous().view(-1, feature_dim)
            
            # 计算相似度矩阵 [batch_size * seq_len, batch_size * seq_len]
            similarity_matrix = F.cosine_similarity(
                ts_flat.unsqueeze(1), img_flat.unsqueeze(0), dim=2
            ) / self.temperature
            
            # 创建标签：同一时间点的样本为正样本
            labels = torch.arange(batch_size * seq_len, device=ts.device)
            
            # 计算对比损失
            loss_ts_to_img = F.cross_entropy(similarity_matrix, labels)
            loss_img_to_ts = F.cross_entropy(similarity_matrix.T, labels)
            
            # 对称的对比损失
            contrastive_loss = (loss_ts_to_img + loss_img_to_ts) / 2
            
            return contrastive_loss
        elif self.contrast_type == 'temporal':
            ts_flat = ts.mean(dim=1)
            img_flat = img.mean(dim=1)
            similarity_matrix = F.cosine_similarity(
                ts_flat.unsqueeze(1), img_flat.unsqueeze(0), dim=2
            ) / self.temperature
            labels = torch.arange(batch_size, device=ts.device)
            loss_ts_to_img = F.cross_entropy(similarity_matrix, labels)
            loss_img_to_ts = F.cross_entropy(similarity_matrix.T, labels)

            # 对称的对比损失
            contrastive_loss = (loss_ts_to_img + loss_img_to_ts) / 2
            return contrastive_loss
        else:
            raise ValueError(f"Invalid contrast_type: {self.contrast_type}")

class causal_kd(nn.Module):
    def __init__(self, temperature=0.1, contrast_type='instance'):
        super(causal_kd, self).__init__()
        self.temperature = temperature
        self.contrast_type = contrast_type
    
    def forward(self, ts, img):
        batch_size, seq_len, feature_dim = ts.shape
        f_h_img = img[:, seq_len//2:]
        l_h_ts = ts[:, :seq_len//2]

        if self.contrast_type == 'instance':
            # Reshape
            f_h_img = f_h_img.reshape(-1, feature_dim)
            l_h_ts = l_h_ts.reshape(-1, feature_dim)

            # Normalize
            f_h_img = F.normalize(f_h_img, dim=1)
            l_h_ts = F.normalize(l_h_ts, dim=1)

            similarity_matrix = torch.matmul(l_h_ts, f_h_img.T)

            # 应用温度系数
            similarity_matrix /= self.temperature

            # 修正：标签长度应该匹配 reshape 后的样本数
            num_samples = batch_size * (seq_len // 2)
            labels = torch.arange(num_samples).to(ts.device)

            # 计算交叉熵损失
            loss_1_to_2 = F.cross_entropy(similarity_matrix, labels)
            loss_2_to_1 = F.cross_entropy(similarity_matrix.T, labels)

            total_loss = (loss_1_to_2 + loss_2_to_1) / 2
            return total_loss
        elif self.contrast_type == 'temporal':
            ts_flat = ts.mean(dim=1)
            img_flat = img.mean(dim=1)
            similarity_matrix = F.cosine_similarity(
                ts_flat.unsqueeze(1), img_flat.unsqueeze(0), dim=2
            ) / self.temperature
            labels = torch.arange(batch_size, device=ts.device)
            loss_ts_to_img = F.cross_entropy(similarity_matrix, labels)
            loss_img_to_ts = F.cross_entropy(similarity_matrix.T, labels)

            # 对称的对比损失
            total_loss = (loss_ts_to_img + loss_img_to_ts) / 2
            return total_loss
        else:
            raise ValueError(f"Invalid contrast_type: {self.contrast_type}")

