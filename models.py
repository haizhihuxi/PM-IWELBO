

import logging
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import Config
import math

logger = logging.getLogger(__name__)
random.seed(42)
torch.manual_seed(42)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 检查 config.n_embd（表示输入嵌入维度的配置参数）是否能够被 config.n_head（表示头数）整除。这是因为在多头自注意力中，每个头的维度都应该相同。
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)  # 线性层768*768
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)  # dropout 神经元
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask因果掩码 只关注输入序列中前面的单词对后面的单词的影响，而不允许后面的单词影响前面的单词
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                             .view(1, 1, config.max_seqlen, config.max_seqlen))
        # 120*120上三角矩阵改为1 x 1 x config.max_seqlen x config.max_seqlen来将批次大小设置为1。 创建一个叫做“mask”的buffer
        self.n_head = config.n_head  # 头的数目8

        # Create an input mask that ignores the last 520 columns of the embedding during evaluation
        self.register_buffer("input_mask",
                             torch.cat([torch.ones(config.n_embd - config.n_auxil), torch.zeros(config.n_auxil)]))

    def forward(self, x, weight, state, weight_num=None):
        B, T, C = x.size()  # 32 120 768
        # B批量大小（batch size），T序列长度（sequence length），self.n_head注意力头的数量，C // self.n_head每个头所处理的特征数量
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B32, nh8, T120, hs96)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 计算点积
        att = q @ k.transpose(-2, -1)

        # 计算缩放因子，即特征维度的平方根的倒数
        scale_factor = 1.0 / math.sqrt(k.size(-1))
        # 应用缩放因子
        att = att * scale_factor  # q(32,8,120,96)与k转置(32,8,96,120)矩阵乘法得(32,8,120,120)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # mask无关位置,mask中为0的位置用浮点数‘-inf’填充
        att = F.softmax(att, dim=-1)  # softmax(32,8,120,120)
        att = self.attn_drop(att)  # dropout 非0参数除以0.9，放大
        y = att @ v  # (B32, nh8, T120, T120) x (B32, nh8, T120, hs96) -> (B32, nh8, T120, hs96) #*V
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # (32,8,120,96)->(32,120,8,96)->无内存碎片，确保连续->(32,120,768)
        # output projection
        y = self.resid_drop(self.proj(y))  # 先通过线形层再drop
        return y


class CausalSelfAttention_newsim_weight(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 检查 config.n_embd（表示输入嵌入维度的配置参数）是否能够被 config.n_head（表示头数）整除。这是因为在多头自注意力中，每个头的维度都应该相同。
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)  # 线性层768*768
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.weights_layer = nn.Linear(config.max_seqlen, config.max_seqlen, bias=False)  # 添加用于学习权重的全连接层
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)  # dropout 神经元
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.weight_concat = nn.Linear(config.max_seqlen, config.max_seqlen)
        # causal mask因果掩码 只关注输入序列中前面的单词对后面的单词的影响，而不允许后面的单词影响前面的单词
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                             .view(1, 1, config.max_seqlen, config.max_seqlen))
        # 120*120上三角矩阵改为1 x 1 x config.max_seqlen x config.max_seqlen来将批次大小设置为1。 创建一个叫做“mask”的buffer
        self.n_head = config.n_head  # 头的数目8
        self.compary_threshold = config.traj_compary_threshold
        self.register_buffer("input_mask",
                             torch.cat([torch.ones(config.n_embd - config.n_auxil), torch.zeros(config.n_auxil)]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(Config.max_seqlen, Config.max_seqlen)
        self.device = "cuda:0"
        
        # 可学习权重缩放因子
        self.v_weight_scale = nn.Parameter(torch.ones(1))  # 初始化为1
        self.att_weight_scale = nn.Parameter(torch.ones(1))  # 初始化为1

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )

    def forward(self, x, weight, state, weight_num=None):
        torch.cuda.empty_cache()
        B, T, C = x.size()  # 32 120 768
        q = self.value(x)
        k = self.value(x)
        v = self.value(x)

        if state:
            # current_torch_seed = torch.initial_seed()
            # print(f"Current PyTorch seed: {current_torch_seed}")
            if random.random() > 0.6:
                noise_scale = 0.1
                weights_row = weight[:, :, 0].unsqueeze(2)
                noise_row = torch.randn_like(weights_row) * weights_row.abs() * noise_scale
                weights_row = weights_row + noise_row


                weight_line = weight[:, :, 0].unsqueeze(1).unsqueeze(2)
                noise_line = torch.randn_like(weight_line) * weight_line.abs() * noise_scale
                weight_line = weight_line + noise_line
            else:
                weights_row = torch.ones_like(v)
                weight_line = torch.ones_like(v)
            # weights_row = weight[:, :, 0].unsqueeze(2)
            # weight_line = weight[:, :, 0].unsqueeze(1).unsqueeze(2)
            # weights_row = torch.ones_like(v)
            # weight_line = torch.ones_like(v)
        else:
            weights_row = torch.ones_like(v)
            weight_line = torch.ones_like(v)

        v = v * weights_row * self.v_weight_scale
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # 计算点积
        att = q @ k.transpose(-2, -1)
        all_ones = (weight_line == 1).all().item()
        if all_ones:
            weight_line = torch.ones_like(att)
        att = att * weight_line * self.att_weight_scale
        scale_factor = 1.0 / math.sqrt(k.size(-1))
        att = att * scale_factor  # q(32,8,120,96)与k转置(32,8,96,120)矩阵乘法得(32,8,120,120)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # mask无关位置,mask中为0的位置用浮点数‘-inf’填充
        att = F.softmax(att, dim=-1)  # softmax(32,8,120,120)
        att = self.attn_drop(att)  # dropout 非0参数除以0.9，放大
        att = att.float()
        y = att @ v  # (B32, nh8, T120, T120) x (B32, nh8, T120, hs96) -> (B32, nh8, T120, hs96) #*V
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))  # 先通过线形层再drop
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化层，将序列信息压缩，用于预测目的地信息
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x, weight, state, des, with_auxiliary, num):
        x = x + self.attn(self.ln1(x), weight, state, num)
        x = x + self.mlp(self.ln2(x))
        # if state and with_auxiliary:
        #     auxil_x = x
        #     auxil_x = self.global_avg_pool(auxil_x.transpose(1, 2)).transpose(1, 2)  # (32,1,768)
        #     des = des + self.mlp_des(self.ln3(auxil_x))
        return x, weight, state, des, with_auxiliary, num


class Block_newsim_weight(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )
        self.attn = CausalSelfAttention_newsim_weight(config)

    def forward(self, x_embedding, weight, state, des, with_auxiliary, num):
        """将x归一化后输入自注意力模块attn再与自身相加，然后再次归一化，并通过前馈神经网络mlp进行处理"""
        x_embedding = x_embedding + self.attn(self.ln1(x_embedding), weight, state, num)
        x_embedding = x_embedding + self.mlp(self.ln2(x_embedding))
        # if state and with_auxiliary:
        #     auxil_x = x_embedding
        #     auxil_x = self.global_avg_pool(auxil_x.transpose(1, 2)).transpose(1, 2)  # (32,1,768)
        #     des = des + self.mlp_des(self.ln3(auxil_x))

        return x_embedding, weight, state, des, with_auxiliary, num


class CustomSequential(nn.Module):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x_embedding, weight, state, des, auxiliary, num):
        # 逐个模块处理输入
        for module in self.modules_list:
            x_embedding, weight, state, des, auxiliary, num = module(x_embedding, weight, state, des, auxiliary, num)
        return x_embedding, weight, state, des, auxiliary, num


class adaptiveModel(nn.Module):
    def __init__(self, config):
        super(adaptiveModel, self).__init__()
        # 创建一个包含所需顺序的块的列表
        layers = []

        for _ in range(0, 2):
            layers.append(Block(config))
        for _ in range(0, 1):
            layers.append(Block_newsim_weight(config))
        for _ in range(0, 5):
            layers.append(Block(config))

        # for _ in range(0, 8):
        #     layers.append(Block(config))

        # 使用列表创建 CustomSequential
        self.blocks = CustomSequential(*layers)

    def forward(self, x_embedding, weight, state, des, auxiliary, num):
        return self.blocks(x_embedding, weight, state, des, auxiliary, num)


class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model=None):
        super().__init__()

        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.lat_target_size = config.lat_target_size
        self.lon_target_size = config.lon_target_size
        self.full_size = config.full_size
        self.full_des_size = config.full_des_size
        self.n_head = config.n_head

        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.n_lat_target_embd = config.n_lat_target_embd
        self.n_lon_target_embd = config.n_lon_target_embd
        self.register_buffer("att_sizes", torch.tensor(
            [config.lat_size, config.lon_size, config.sog_size, config.cog_size]))  # 250,270,30,72
        self.register_buffer("destination_sizes", torch.tensor([config.lat_target_size, config.lon_target_size]))
        self.register_buffer("emb_sizes", torch.tensor(
            [config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd, config.n_lat_target_embd,
             config.n_lon_target_embd]))

        if hasattr(config, "partition_mode"):  # 数据分区的模型
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model

        if hasattr(config, "lat_min"):  # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max - config.lat_min
            self.lon_range = config.lon_max - config.lon_min
            self.sog_range = 30.

        if hasattr(config, "mode"):  # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)  # 元素的数量250  每个元素在嵌入层中的表示的维度256
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)  # 270 256
        self.lat_target_emb = nn.Embedding(self.lat_target_size, config.n_lat_target_embd)
        self.lon_target_emb = nn.Embedding(self.lon_target_size, config.n_lon_target_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)  # 30 128
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)  # 72 128
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)  # 0.1

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.ln_auxil = nn.LayerNorm(config.n_auxil)
        if self.mode in ("mlp_pos", "mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.head_auxil = nn.Linear(config.n_auxil, config.n_auxil, bias=False)
        else:
            # 将概率转化为离散经纬度的
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False)  # Classification head
            self.head_auxil = nn.Linear(config.n_auxil, self.full_des_size, bias=False)
        self.max_seqlen = config.max_seqlen
        self.blocks = adaptiveModel(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 将输入数据的维度扩大四倍 进一步提取特征加速训练
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.mlp_des = nn.Sequential(
            nn.Linear(config.n_embd, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.n_auxil),
        )

        # 计算模型中所有参数的数量
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def print_param_names(model):
        for name, param in model.named_parameters():
            print(name)

    # 初始化模型权重
    # 对于线性层和嵌入层，使用正态分布来初始化权重，对于归一化层，将偏置项初始化为 0，并将权重初始化为 1
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)  # 均值：0  标准差：0.02
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        weight_decay_params = set()
        no_weight_decay_params = set()

        decay_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.AdaptiveAvgPool2d, torch.nn.Conv2d)
        no_decay_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.GroupNorm)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = f'{module_name}.{param_name}' if module_name else param_name

                if param_name.endswith('bias'):
                    no_weight_decay_params.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, decay_modules):
                    weight_decay_params.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, no_decay_modules):
                    no_weight_decay_params.add(full_param_name)
                elif 'att_weight_scale' in full_param_name or 'v_weight_scale' in full_param_name:
                    # 特殊处理标量参数，添加到 no_weight_decay
                    no_weight_decay_params.add(full_param_name)

        no_weight_decay_params.add('pos_emb')

        param_dict = {param_name: param for param_name, param in self.named_parameters() if param.requires_grad}
        intersect_params = weight_decay_params & no_weight_decay_params
        all_params = weight_decay_params | no_weight_decay_params

        assert len(intersect_params) == 0, "参数 %s 同时出现在了 decay 和 no_decay 集合中!" % (str(intersect_params),)
        assert len(param_dict.keys() - all_params) == 0, \
            "参数 %s 没有被分配到 decay 或 no_decay 集合中!" % (str(param_dict.keys() - all_params),)

        missing_params_no = [param_name for param_name in no_weight_decay_params if param_name not in param_dict]
        missing_params = [param_name for param_name in weight_decay_params if param_name not in param_dict]
        if missing_params:
            print("未找到的衰减参数：", missing_params)
            print("未找到的不衰减参数：", missing_params_no)

        optimizer_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(list(weight_decay_params)) if
                           param_name in param_dict],
                "weight_decay": train_config.weight_decay
            },
            {
                "params": [param_dict[param_name] for param_name in sorted(list(no_weight_decay_params)) if
                           param_name in param_dict],
                "weight_decay": 0.0
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


    def to_indexes(self, x, destination=None):  # uniform
        idxs = (x * self.att_sizes).long()  # 真实数据（lan，lon,sog,cog）*[250, 270,  30,  72] 每个真实值都是一个独一无二的索引
        if destination is not None:
            device = 'cuda:0'
            destination = destination.to(device)
            idxs_des = (destination * self.destination_sizes).long()
        else:
            idxs_des = None
        return idxs, idxs_des

    # def smoothness_loss_real_position(self, lat_logits, lon_logits):
    #     # 获取概率最高的类别索引
    #     lat_indices = torch.argmax(lat_logits, dim=-1)  # (batchsize, seqlen)
    #     lon_indices = torch.argmax(lon_logits, dim=-1)  # (batchsize, seqlen)
    #
    #     # 计算真实的纬度和经度位置
    #     # lat_real_positions = lat_indices * 250  # 乘以缩放因子 250
    #     # lon_real_positions = lon_indices * 270  # 乘以缩放因子 270
    #
    #     # 计算相邻时间步的差异 (纬度和平滑损失)
    #     lat_diffs = lat_indices[:, 1:] - lat_indices[:, :-1]
    #     lon_diffs = lon_indices[:, 1:] - lon_indices[:, :-1]
    #
    #     # 计算差异的平方和
    #     lat_smooth_loss = torch.sum(lat_diffs ** 2)
    #     lon_smooth_loss = torch.sum(lon_diffs ** 2)
    #
    #     # 返回纬度和经度的平滑损失
    #     return lat_smooth_loss, lon_smooth_loss

    def forward(self, x, destination, seq_weight=None, num=None, masks=None,
                weight_caculate=False, with_auxiliary=True):  # model根据参数，直接加载forward函数

        idxs, idxs_des = self.to_indexes(x, destination)
        inputs = idxs[:, :-1, :].contiguous()  # 输入排除最后一行
        targets = idxs[:, 1:, :].contiguous()  # 目标值排除第一行
        batchsize, seqlen, _ = inputs.size()  # 32 120
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."  # 检查seqlen（输入序列的长度）是否小于或等于self.max_seqlen（模型的最大序列长度）。

        # forward the model
        lat_embeddings = self.lat_emb(
            inputs[:, :, 0])  # (bs, seqlen) -> (bs, seqlen, lat_size) e.t. (32,120)->(32,# 120,256)词嵌入
        lon_embeddings = self.lon_emb(inputs[:, :, 1])  # (bs, seqlen, lon_size)
        sog_embeddings = self.sog_emb(inputs[:, :, 2])
        cog_embeddings = self.cog_emb(inputs[:, :, 3])
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),
                                     dim=-1)  # 4-hot 在最后一个维度拼接
        position_embeddings = self.pos_emb[:, :seqlen, :]
        fea = self.drop(token_embeddings + position_embeddings)
        # 初始化一个空的张量
        batch_size = x.shape[0]
        third_dim = 512  # 自定义的第三维度大小

        # 创建新的张量
        des = torch.zeros(batch_size, 1, third_dim, device=x.device, dtype=x.dtype)
        if seq_weight is not None:
            seq_weight = seq_weight[:, :-1, :]

        fea = self.blocks(fea, seq_weight, weight_caculate, des, with_auxiliary, num)
        fea_main = fea[0]  # 主任务
        fea_main = self.ln_f(fea_main)  # (bs, seqlen, n_embd) e.t. (32,120,768)归一化层
        logits = self.head(fea_main)  # (bs, seqlen, full_size)e,t. (32,120,622) or (bs, seqlen, n_embd)

        # if with_auxiliary:
        #     fea_auxil = fea[3] / 8  # 辅助任务
        #     fea_auxil = self.ln_auxil(fea_auxil)
        #     logits_auxil = self.head_auxil(fea_auxil)
        #     lat_logits_target, lon_logits_target = torch.split(logits_auxil,
        #                                                        (self.lat_target_size, self.lon_target_size),
        #                                                        dim=-1)
        # 将经过Layer Normalization处理后的特征图通过一个全连接层（即self.head）进行提取特征并分类。这里的输出logits是模型对于输入的预测结果。
        lat_logits, lon_logits, sog_logits, cog_logits = torch.split(logits, (
            self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)

        # Calculate the loss
        loss = None
        if targets is not None:
            # 交叉熵损失函数

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), targets[:, :, 2].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            # 交叉熵只计算目标索引对应位置的概率值的负对数作为顺势函数的输出
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size),
                                       targets[:, :, 3].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size),
                                       targets[:, :, 0].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size),
                                       targets[:, :, 1].view(-1),
                                       reduction="none").view(batchsize, seqlen)
            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)
            if masks is not None:
                # 使用掩码（masks）进行归一化。
                loss = loss * masks
                loss = loss.sum(dim=1) / masks.sum(dim=1)  # 取每个序列长度的损失
            loss = loss.mean()  # 一个值

            # if with_auxiliary:
            #     lat_target_loss = F.cross_entropy(lat_logits_target.view(-1, self.lat_target_size),
            #                                       idxs_des[:, :, 0].view(-1),
            #                                       reduction="none").view(batchsize, -1)
            #     lon_target_loss = F.cross_entropy(lon_logits_target.view(-1, self.lon_size),
            #                                       idxs_des[:, :, 1].view(-1),
            #                                       reduction="none").view(batchsize, -1)
            #
            #     loss_tuple_des = (lat_target_loss, lon_target_loss)
            #     loss_des = sum(loss_tuple_des)
            #     loss_des = loss_des.mean()
            #     loss += loss_des



        return logits, loss
