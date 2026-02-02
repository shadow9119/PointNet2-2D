'''
Modified by Yiwen Lin
Date: Jul 2023

Modified by Sitong Chen
Date: Oct 2024
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module): # get_model的父类是nn.Module
    # 类初始化 (__init__ 方法)
    def __init__(self, num_classes, conf_channel=False):
        super(get_model, self).__init__()  # 调用 get_model 类的父类的初始化方法
        if conf_channel:
            additional_channel = 1
        else:
            additional_channel = 0
        self.conf_channel = conf_channel
        # SA：set abstraction 点集抽取模块：得到采样点的多个半径的局部特征
        # 用于局部点云特征抽取，sa1, sa2, 和 sa3 是不同层次的抽取模块。
        # 首先通过sample_and_group的操作形成局部的group，然后对局部的group中的每一个点做MLP操作，最后进行局部的最大池化，得到局部的全局特征。
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [32, 64, 128], 2+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])  # 这里2+addtional_channel, 2就是修改后的，表示二维坐标
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 2, mlp=[256, 512, 1024], group_all=True) # 这里512 + 2是修改后的，表示二维坐标
        # FP：feature propagation 特征传播模块（跨层连接，多个全连接）
        # 用于特征传播的模块，fp1, fp2, 和 fp3 是不同层次的传播模块。
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])  # in_channel = 1024+256+256 (sa3_out+sa2_out)
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])  # in_channel = 256+128+128+64 (fp3_out+sa1_out)
        self.fp1 = PointNetFeaturePropagation(in_channel=133+additional_channel, mlp=[128, 128]) # in_channel = 128+1+3+3+additional_channel, previous: 128+1+2+2+additional_channel
        # Conv1d, BatchNorm1d, Dropout: 传统的卷积层、批归一化层和 dropout 层，用于最终特征的处理。
        self.conv1 = nn.Conv1d(128, 128, 1)  # 一维卷积层，输入通道数128，输出通道数128，卷积核大小为1 → 时间维度上，单点卷积
        self.bn1 = nn.BatchNorm1d(128) # 一维批归一化层，输入通道数128=conv1输出通道数
        self.drop1 = nn.Dropout(0.5) # 正则化模型，防止过拟合，每个输入元素有 50% 的概率被丢弃（即置为 0）
        self.conv2 = nn.Conv1d(128, num_classes, 1) # 一维卷积层，输入通道数128，输出通道数num_classes，卷积核大小是 1，意味着每个时间步的特征都被独立地映射到最终的类别数上。

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.conf_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:2,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B,128+128+64,512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,256+256,128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B,1024,1]
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B,256,128]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B,128,512]
        cls_label_one_hot = cls_label.view(B,1,1).repeat(1,1,N)  # previously (B,16,1) since there were 16 classes; input is [B,1], repeat to [B,1.N]
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)  # [B,128,N]
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))  # [B,128,N]
        x = self.drop1(feat)
        x = self.conv2(x)  # [B,num_classes,N]
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # [B,N,num_classes]
        return x, l3_points

# 计算负对数似然损失（Negative Log Likelihood Loss）
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weight=None):
        total_loss = F.nll_loss(pred, target, weight=weight)  # 负对数似然损失函数，计算的是预测概率的对数损失，weight是不同类别的损失加权，损失值越小，表示模型的预测与真实标签越接近

        return total_loss

# 简单Focal loss函数
class FocalLoss(nn.Module):
    '''
    Multi-class Focal loss implementation
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = input
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

# 自适应Focal loss函数
class AdaptiveFocalLoss(nn.Module):
    """
    自适应 Focal Loss
    适配模型的 Log_Softmax 输出
    """
    def __init__(self, gamma=2.0, eps=1e-8):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        """
        pred: [B*N, C] 模型输出的对数概率 (Log Softmax output)
        target: [B*N] 真实标签
        """
        # 1. 还原概率 (因为输入 pred 已经是 log(p) 了，所以 p = exp(pred))
        probs = torch.exp(pred)
        
        # 2. 动态计算 alpha (基于当前 Batch 的类别分布)
        # 注意：这里假设是二分类或多分类，以下逻辑通用
        # 获取当前 batch 中每个类别的数量
        C = pred.shape[1]
        
        # 初始化 alpha
        alpha = torch.zeros_like(probs)
        
        # 计算总样本数
        total_count = target.numel() + self.eps
        
        # 针对每个类别计算反向频率权重
        for c in range(C):
            # 统计类别 c 的样本数
            count_c = (target == c).sum().float()
            # 简单反向频率：类别越少，权重越大
            # 权重 = (总数 - 该类数量) / 总数
            # 对于二分类：类0权重 = 类1数量占比；类1权重 = 类0数量占比
            weight_c = (total_count - count_c) / total_count
            
            # 将该权重赋予对应的列
            alpha[:, c] = weight_c

        # 3. 获取目标类别的概率 pt 和权重 at
        # gather 用于提取 target 对应位置的概率和权重
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        at = alpha.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 获取目标类别的 log_pt (直接从 pred 取，避免重复 log 运算导致数值不稳定)
        log_pt = pred.gather(1, target.unsqueeze(1)).squeeze(1)

        # 4. 计算 Focal Loss
        # 公式: -alpha * (1 - pt)^gamma * log(pt)
        loss = -at * (1 - pt) ** self.gamma * log_pt

        return loss.mean()
