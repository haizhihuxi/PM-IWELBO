

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):

        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):

        V = self.l_data[idx]
        m_v = V["traj"][:,:4]# lat, lon, sog, cog前4列
        #         m_v[m_v==1] = 0.9999
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)  # 返回这两个长度中的较小值，序列的实际长度

        target = np.full((1 , 2), np.nan)#创建目的地序列，将最轨迹终点填入
        target[:, 0] = m_v[seqlen - 1][0]
        target[:, 1] = m_v[seqlen - 1][1]
        # m_v = np.hstack([m_v, target])

        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)  # 转化为torch.tensor

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.  # 表示这些位置是非padding

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)

        return seq, target, mask, seqlen, mmsi, time_start


class AISDataset_new(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):

        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):

        V = self.l_data[idx]
        m_v = V["traj"][:, :4]  # lat, lon, sog, cog前4列
        weight_v = V["traj"][:, 6:]
        #         m_v[m_v==1] = 0.9999
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)  # 返回这两个长度中的较小值，序列的实际长度

        target = np.full((1, 2), np.nan)  # 创建目的地序列，将最轨迹终点填入
        target[:, 0] = m_v[seqlen - 1][0]
        target[:, 1] = m_v[seqlen - 1][1]

        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :] # (141,4)
        seq = torch.tensor(seq, dtype=torch.float32)  # 转化为torch.tensor

        seq_weight = np.zeros((self.max_seqlen, 5))
        seq_weight[:seqlen, :] = weight_v[:seqlen, :]  # (141,5)

        # 填充seq_weight后续为0的位置
        last_non_zero_weight = seq_weight[seqlen - 1]
        if seqlen < self.max_seqlen:
            for i in range(seqlen, self.max_seqlen):
                seq_weight[i] = last_non_zero_weight
        seq_weight = torch.tensor(seq_weight, dtype=torch.float32)  # 转化为torch.tensor

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.  # 表示这些位置是非padding

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        # time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)
        # time_start = V["traj"][0, 4].clone().detach().int()
        return seq, target, mask, seqlen, mmsi, seq_weight

class AISDataset_new(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):

        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):

        V = self.l_data[idx]
        m_v = V["traj"][:, :4]  # lat, lon, sog, cog前4列
        weight_v = V["traj"][:, 6:]
        #         m_v[m_v==1] = 0.9999
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)  # 返回这两个长度中的较小值，序列的实际长度

        target = np.full((1, 2), np.nan)  # 创建目的地序列，将最轨迹终点填入
        target[:, 0] = m_v[seqlen - 1][0]
        target[:, 1] = m_v[seqlen - 1][1]

        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :] # (141,4)
        seq = torch.tensor(seq, dtype=torch.float32)  # 转化为torch.tensor

        seq_weight = np.zeros((self.max_seqlen, 5))
        seq_weight[:seqlen, :] = weight_v[:seqlen, :]  # (141,5)

        # 填充seq_weight后续为0的位置
        last_non_zero_weight = seq_weight[seqlen - 1]
        if seqlen < self.max_seqlen:
            for i in range(seqlen, self.max_seqlen):
                seq_weight[i] = last_non_zero_weight
        seq_weight = torch.tensor(seq_weight, dtype=torch.float32)  # 转化为torch.tensor

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.  # 表示这些位置是非padding

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        # time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)
        # time_start = V["traj"][0, 4].clone().detach().int()
        return seq, target, mask, seqlen, mmsi, seq_weight




