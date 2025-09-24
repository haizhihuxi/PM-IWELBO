
import numpy as np
import os
import math
import logging
import random
import datetime
import socket


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.pi = torch.acos(torch.zeros(1)).item()*2


#初始化随机数种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)#为CPU中设置种子，生成随机数
    torch.cuda.manual_seed_all(seed)#为特定GPU设置种子，生成随机数
    torch.backends.cudnn.deterministic = True#每次返回卷积算法是确定的，即默认算法

    
def new_log(logdir,filename):
    """定义日志格式
    """
    filename = os.path.join(logdir,
                            datetime.datetime.now().strftime("log_%Y-%m-%d-%H-%M-%S_"+socket.gethostname()+"_"+filename+".log"))
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        format="%(asctime)s - %(name)s - %(message)s",
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)   
    
def haversine(input_coords, 
               pred_coords):
    """ Calculate the haversine distances between input_coords and pred_coords.
    计算输入坐标（input_coords）和预测坐标（pred_coords）之间的哈弗赛恩距离
    
    Args:
        input_coords, pred_coords: Tensors of size (...,N), with (...,0) and (...,1) are
        the latitude and longitude in radians.
    
    Returns:
        The havesine distances between
    """
    R = 6371
    lat_errors = pred_coords[...,0] - input_coords[...,0]
    lon_errors = pred_coords[...,1] - input_coords[...,1]#预测坐标减去输入坐标
    a = torch.sin(lat_errors/2)**2\
        +torch.cos(input_coords[:,:,0])*torch.cos(pred_coords[:,:,0])*torch.sin(lon_errors/2)**2
    c = 2*torch.atan2(torch.sqrt(a),torch.sqrt(1-a))
    d = R*c
    return d

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)#函数使用torch.topk(logits, k)来获取logits中的最高k个值以及它们的索引。这些值和索引会被存储在变量v和ix中。
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    #函数创建了一个logits的克隆，并将所有小于最高k个值中的最低值的元素设置为负无穷大。
    # 这实际上是在为预测的类别设置一个阈值，低于这个阈值的预测将被排除。
    return out

def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """Keep only k values nearest the current idx.仅保留最接近当前 idx 的 k 值。
    
    Args:
        att_logits: a Tensor of shape (bachsize, data_size). 
        att_idxs: a Tensor of shape (bachsize, 1), indicates the current idxs.表示当前索引号
        r_vicinity: number of values to be kept.要保留的值的个数
    """
    device = att_logits.device
    idx_range = torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0],1)
    #idx_range 是一个二维张量，其形状与 att_logits 相同，里面的元素是从0开始的连续整数。这个张量可以用来标记 att_logits 中的每一个元素
    idx_dists = torch.abs(idx_range - att_idxs)#计算绝对差值
    out = att_logits.clone()
    out[idx_dists >= r_vicinity/2] = -float('Inf')#将 out 中所有距离大于等于 r_vicinity/2 的元素的 logits 设置为负无穷。
    # 这实际上是在创建一个 "软" 掩码，使得距离大于 r_vicinity/2 的所有元素在计算损失时被忽略（因为它们的 logits 为负无穷）。
    return out

def kendall_tau(x, y):
    """
    计算两个序列的 Kendall's Tau.
    Args:
    - x (array-like): 第一个序列
    - y (array-like): 第二个序列
    Returns:
    - tau (float): Kendall's Tau 系数
    """
    assert len(x) == len(y), "The lengths of both arrays must be equal"
    n = len(x)
    num_concordant = 0
    num_discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] < x[j] and y[i] < y[j]) or (x[i] > x[j] and y[i] > y[j]):
                num_concordant += 1
            elif (x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j]):
                num_discordant += 1

    tau = (num_concordant - num_discordant) / (0.5 * n * (n - 1))
    return tau


# 定义将序列转换为排名序列的函数
def rankdata(a):
    n = len(a)
    ivec = np.argsort(a)
    svec = np.argsort(ivec)
    rvec = np.zeros(n, dtype=int)
    for i in range(n):
        rvec[svec[i]] = i
    return rvec
