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


class inter_causal_kd(nn.Module):
    def __init__(self, temperature=0.1, contrast_type='instance'):
        super(inter_causal_kd, self).__init__()
        self.temperature = temperature
        self.contrast_type = contrast_type

    def forward(self, ts, img, perturb_img):
        # ts, img, perturb_img: [batch_size, seq_len, feature_dim]
        batch_size, seq_len, feature_dim = ts.shape

        if self.contrast_type == 'instance':
            # 展平所有特征
            ts_flat = ts.reshape(-1, feature_dim)  # [batch_size * seq_len, feature_dim]
            img_flat = img.reshape(-1, feature_dim)  # [batch_size * seq_len, feature_dim]
            perturb_img_flat = perturb_img.reshape(-1, feature_dim)  # [batch_size * seq_len, feature_dim]
            
            # L2归一化
            ts_flat = F.normalize(ts_flat, dim=1)
            img_flat = F.normalize(img_flat, dim=1)
            perturb_img_flat = F.normalize(perturb_img_flat, dim=1)
            
            # 计算相似度
            # 正对：ts与img的相似度 [batch_size * seq_len, batch_size * seq_len]
            pos_similarity = torch.matmul(ts_flat, img_flat.T) / self.temperature
            
            # 负对：ts与perturb_img的相似度 [batch_size * seq_len, batch_size * seq_len]
            neg_similarity = torch.matmul(ts_flat, perturb_img_flat.T) / self.temperature
            
            # 构建对比学习的logits：[N, 2*N]，其中前N个是正样本，后N个是负样本
            num_samples = batch_size * seq_len
            logits = torch.cat([pos_similarity, neg_similarity], dim=1)
            
            # 标签：对于每个样本，对应的正样本在前N个位置中
            labels = torch.arange(num_samples, device=ts.device)
            
            # 计算对比损失
            contrastive_loss = F.cross_entropy(logits, labels)
            
            return contrastive_loss
            
        elif self.contrast_type == 'temporal':
            # 在时间维度上平均池化
            ts_pooled = ts.mean(dim=1)  # [batch_size, feature_dim]
            img_pooled = img.mean(dim=1)  # [batch_size, feature_dim]
            perturb_img_pooled = perturb_img.mean(dim=1)  # [batch_size, feature_dim]
            
            # L2归一化
            ts_pooled = F.normalize(ts_pooled, dim=1)
            img_pooled = F.normalize(img_pooled, dim=1)
            perturb_img_pooled = F.normalize(perturb_img_pooled, dim=1)
            
            # 计算相似度
            # 正对：ts与img的相似度 [batch_size, batch_size]
            pos_similarity = torch.matmul(ts_pooled, img_pooled.T) / self.temperature
            
            # 负对：ts与perturb_img的相似度 [batch_size, batch_size]
            neg_similarity = torch.matmul(ts_pooled, perturb_img_pooled.T) / self.temperature
            
            # 构建对比学习的logits：[batch_size, 2*batch_size]
            logits = torch.cat([pos_similarity, neg_similarity], dim=1)
            
            # 标签：对于每个样本，对应的正样本在前batch_size个位置中
            labels = torch.arange(batch_size, device=ts.device)
            
            # 计算对比损失
            contrastive_loss = F.cross_entropy(logits, labels)
            
            return contrastive_loss
            
        else:
            raise ValueError(f"Invalid contrast_type: {self.contrast_type}")

class patch_kd(nn.Module):
    def __init__(self, temperature=0.1, patch_len_threshold=48):
        super(patch_kd, self).__init__()
        self.temperature = temperature
        self.patch_len_threshold = patch_len_threshold
        self.kl_loss = nn.KLDivLoss(reduction='none')
        
    def create_patches(self, x, patch_len, stride):
        """
        创建时间序列的patches
        Args:
            x: [B, L, C] 输入时间序列
            patch_len: patch长度
            stride: 滑动步长
        Returns:
            patches: [B, C, num_patches, patch_len]
        """
        x = x.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
        B, C, L = x.shape
        
        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)
        
        return patches

    def fouriour_based_adaptive_patching(self, img, ts):
        """
        基于傅里叶变换的自适应分块
        Args:
            img: [B, L, C] 教师模型输出（图像特征）
            ts: [B, L, C] 学生模型输出（时序特征）
        Returns:
            img_patch: [B, C, num_patches, patch_len] 图像patches
            ts_patch: [B, C, num_patches, patch_len] 时序patches
        """
        # 使用教师模型输出来确定主导频率
        img_fft = torch.fft.rfft(img, dim=1)
        frequency_list = torch.abs(img_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0  # 忽略DC分量
        top_index = torch.argmax(frequency_list)
        
        # 计算周期和patch参数
        period = (img.shape[1] // top_index) if top_index > 0 else img.shape[1]
        patch_len = min(period // 2, self.patch_len_threshold)
        patch_len = max(patch_len, 4)  # 确保最小patch长度
        stride = patch_len // 2
        
        # 创建patches
        img_patch = self.create_patches(img, patch_len, stride=stride)
        ts_patch = self.create_patches(ts, patch_len, stride=stride)

        return img_patch, ts_patch
    
    def patch_wise_structural_loss(self, true_patch, pred_patch):
        """
        计算patch级别的结构损失
        Args:
            true_patch: [B, C, num_patches, patch_len] 教师模型patches
            pred_patch: [B, C, num_patches, patch_len] 学生模型patches
        Returns:
            linear_corr_loss: 线性相关损失
            var_loss: 方差损失
            mean_loss: 均值损失
        """
        # 计算均值
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        
        # 计算方差和标准差
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var + 1e-8)
        pred_patch_std = torch.sqrt(pred_patch_var + 1e-8)
        
        # 计算协方差
        true_pred_patch_cov = torch.mean(
            (true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), 
            dim=-1, keepdim=True
        )
        
        # 1. 线性相关损失
        patch_linear_corr = (true_pred_patch_cov + 1e-8) / (true_patch_std * pred_patch_std + 1e-8)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. 分布差异损失（使用KL散度）
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
        
        # 3. 均值损失
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()
        
        return linear_corr_loss, var_loss, mean_loss
    
    def gradient_based_dynamic_weighting(self, linear_corr_loss, var_loss, mean_loss, img, ts):
        """
        基于梯度的动态权重计算
        Args:
            linear_corr_loss: 线性相关损失
            var_loss: 方差损失
            mean_loss: 均值损失
            img: 教师模型输出
            ts: 学生模型输出
        Returns:
            alpha, beta, gamma: 动态权重
        """
        # 计算全局相似度特征
        img_mean = torch.mean(img, dim=1, keepdim=True)
        ts_mean = torch.mean(ts, dim=1, keepdim=True)
        img_var = torch.var(img, dim=1, keepdim=True, unbiased=False)
        ts_var = torch.var(ts, dim=1, keepdim=True, unbiased=False)
        img_std = torch.sqrt(img_var + 1e-8)
        ts_std = torch.sqrt(ts_var + 1e-8)
        
        # 计算线性相似度和方差相似度
        covariance = torch.mean((img - img_mean) * (ts - ts_mean), dim=1, keepdim=True)
        linear_sim = (covariance + 1e-8) / (img_std * ts_std + 1e-8)
        linear_sim = (1.0 + linear_sim) * 0.5  # 归一化到[0,1]
        
        var_sim = (2 * img_std * ts_std + 1e-8) / (img_var + ts_var + 1e-8)
        
        # 基于相似度的自适应权重
        # 当线性相关性低时，增加线性相关损失的权重
        alpha = 2.0 - linear_sim.mean().detach()
        
        # 当方差差异大时，增加方差损失的权重
        beta = 2.0 - var_sim.mean().detach()
        
        # 均值权重基于整体相似度
        gamma = (linear_sim * var_sim).mean().detach()
        
        return alpha, beta, gamma

    def forward(self, ts, img):
        """
        前向传播
        Args:
            ts: [B, L, C] 学生模型输出
            img: [B, L, C] 教师模型输出
        Returns:
            patch_loss: patch级别的知识蒸馏损失
        """
        # 自适应分块
        img_patch, ts_patch = self.fouriour_based_adaptive_patching(img, ts)
        
        # 计算patch级别的结构损失
        linear_corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(img_patch, ts_patch)
        
        # 动态权重计算
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(
            linear_corr_loss, var_loss, mean_loss, img, ts
        )

        # 最终的patch知识蒸馏损失
        patch_loss = alpha * linear_corr_loss + beta * var_loss + gamma * mean_loss
        
        return patch_loss