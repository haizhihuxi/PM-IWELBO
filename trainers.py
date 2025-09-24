

import os
import math
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils

from config import Config
import torch
from torch.profiler import profile, record_function, ProfilerActivity

logger = logging.getLogger(__name__)


@torch.no_grad()  # PyTorch的装饰器，推理过程中关闭梯度计算，这可以提高运行速度并减少内存使用。
# 它首先通过模型对输入的序列数据进行预测，得到一系列的logits（预测结果）。
# 然后，根据不同的采样模式和其他参数，对logits进行处理。
# 接着，它会将这些logits转换为概率分布，然后根据这个概率分布进行采样。
# 最后，它会将采样的结果拼接到原始的序列数据后面，并返回更新后的序列数据。
def sample(model,
           target,
           seqs,
           steps,
           temperature=1.0,
           sample=False,  # 采用最大似然估计
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):
    # 测试集采样方式
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. 
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()  # 评估模式，dropout和batch normalization层都将被关闭，这对于预测或测试新数据非常有用。
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed裁剪上下文序列

        # logits.shape: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond, target)  # 获得预测概率
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5  # (batchsize,4)（7，4）在最后采样结果加0.5，防止0出现在分子

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        # lat_logits, lon_logits, sog_logits, cog_logits, lat_logits_target, lon_logits_target = torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size, model.lat_target_size, model.lon_target_size), dim=-1)
        lat_logits, lon_logits, sog_logits, cog_logits = torch.split(logits, (
            model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)
        # 沿着最后一个维度拆分（7，250）（7，270）（7，30）（7，72）

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])  # 真实值乘以250.270.30.72获得索引  idxs_uniform（7，1，4）
            lat_idxs, lon_idxs = idxs[:, 0, 0:1], idxs[:, 0,
                                                  1:2]  # lat_idxs([[ 69],[ 87], [ 42],[206],[ 62],[ 66],[191]])（7，1）
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs,
                                                 r_vicinity)  # 将lat_idxs位置的前后10个元素保留，其余位置设为-inf(7,250)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)  # 将上一步的20个元素保留最大的10个，其余设置为-inf (7,250)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)
            # lat_logits_target = utils.top_k_logits(lat_logits_target, top_k)
            # lon_logits_target = utils.top_k_logits(lon_logits_target, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)  # 获得余下的10个元素的softmax分类结果(7,250)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)
        # lat_probs_target = F.softmax(lat_logits_target, dim=-1)
        # lon_probs_target = F.softmax(lon_logits_target, dim=-1)

        # sample from the distribution or take the most likely
        # 如果sample为True，则使用multinomial方法从概率分布中采样。否则，使用top-k方法选择最可能的类别。
        if sample:
            lat_ix = torch.multinomial(lat_probs,
                                       num_samples=1)  # (batch_size, 1)随机采样一个结果，获得索引tensor([[ 67],[ 85],[ 45],[201],[ 60],[ 67],[189]], device='cuda:0')
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
            # lat_target_ix = torch.multinomial(lat_probs_target, num_samples=1)
            # lon_target_ix = torch.multinomial(lon_probs_target, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)
            # _, lat_target_ix = torch.topk(lat_probs_target, k=1, dim=-1)
            # _, lon_target_ix = torch.topk(lon_probs_target, k=1, dim=-1)

        # ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix, lat_target_ix, lon_target_ix), dim=-1)
        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes  # model中to_index乘以768，这里等比例缩小除以768     (7,1,4)

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)  # 将预测结果添加到原始序列中，拼接到尾端

    return seqs  # (7,18,4)->(7,19,4)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, with_auxiliary=True,
                 device=torch.device("cpu"), aisdls={},
                 INIT_SEQLEN=0):
        self.config = config
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.savedir = savedir
        self.att_sizes = torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size])
        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN
        self.with_auxiliary = with_auxiliary

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)保存最佳模型
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def worker_init_fn(self, worker_id):
        # 获取当前进程的随机数种子
        seed = torch.initial_seed()  # 获取当前worker的种子
        print(f"Worker {worker_id} initial seed: {seed}")

    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'

            model.train(is_train)  # 是否是训练模式，否则验证模式
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            # 训练模式创建进度条
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            batch_means = []
            batch_stds = []

            if is_train:
                for it, (seqs, target, masks, seqlens, mmsis, seq_weight) in pbar:

                    # place data on the correct device
                    idxs = (seqs * self.att_sizes).long()
                    seqs = seqs.to(self.device)
                    target = target.to(self.device)
                    targets = idxs[:, 1:, :].contiguous().to(self.device)
                    seq_weight = seq_weight.to(self.device)
                    masks = masks[:, :-1].to(self.device)

                    # with torch.set_grad_enabled(is_train):
                    #     num = 0
                    #     logits, loss = model(seqs,
                    #                          target, seq_weight, num,
                    #                          masks=masks,
                    #                          weight_caculate=is_train,
                    #                          with_auxiliary=self.with_auxiliary)
                    #     print(loss)
                    #
                    #     loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    #     losses.append(loss.item())
                    #     d_loss += loss.item() * seqs.shape[0]
                    #     d_n += seqs.shape[0]
                    #     model.zero_grad()
                    #     loss.backward()
                    #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    #     optimizer.step()

                    # forward the model for each branch and calculate losses
                    with torch.set_grad_enabled(is_train):
                        # 获取4次前向传播的损失
                        losses_per_branch = []
                        logits_all = []

                        for num in range(5):
                            logits, loss = model(seqs, target, seq_weight, num, masks=masks,
                                                 weight_caculate=is_train, with_auxiliary=self.with_auxiliary)
                            losses_per_branch.append(loss)
                            logits_all.append(logits)
                            print(f"losses_per_branch:{losses_per_branch}")

                        # 初始化加权后的梯度为 0
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad = torch.zeros_like(param.grad)

                        # 计算每个正确类别的概率总和
                        total_lat_prob, total_lon_prob, total_sog_prob, total_cog_prob = 0, 0, 0, 0
                        for i in range(5):
                            lat_logits, lon_logits, sog_logits, cog_logits = torch.split(logits_all[i], (
                                self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
                            # 正确类别
                            total_lat_prob += F.softmax(lat_logits, dim=-1).gather(-1, targets[:, :, 0].unsqueeze(
                                -1)).squeeze(-1)
                            total_lon_prob += F.softmax(lon_logits, dim=-1).gather(-1, targets[:, :, 1].unsqueeze(
                                -1)).squeeze(-1)
                            total_sog_prob += F.softmax(sog_logits, dim=-1).gather(-1, targets[:, :, 2].unsqueeze(
                                -1)).squeeze(-1)
                            total_cog_prob += F.softmax(cog_logits, dim=-1).gather(-1, targets[:, :, 3].unsqueeze(
                                -1)).squeeze(-1)

                            #logits
                            # total_lat_prob += torch.softmax(lat_logits, dim=2)
                            # total_lon_prob += torch.softmax(lon_logits, dim=2)
                            # total_sog_prob += torch.softmax(sog_logits, dim=2)
                            # total_cog_prob += torch.softmax(cog_logits, dim=2)

                        optimizer.zero_grad()
                        # 初始化梯度累积容器
                        grad_accumulator = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

                        for i, loss in enumerate(losses_per_branch):
                            loss = loss.mean()
                            losses.append(loss.item())
                            d_loss += loss.item() * seqs.shape[0]
                            d_n += seqs.shape[0]

                            # print(f"Step {i}: After backward - Gradients:")

                            # 计算当前分支梯度
                            loss.backward(retain_graph=True)  # 保留计算图
                            # for name, param in model.named_parameters():
                            #     if name == "blocks.blocks.modules_list.4.mlp.2.weight":
                            #         print(f"  Before accumulation - Gradient for {name}: {param.grad}")

                            # 计算权重
                            lat_logits, lon_logits, sog_logits, cog_logits = torch.split(logits_all[i], (
                                self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)

                            # 用正确类别的概率
                            lat_prob = F.softmax(lat_logits, dim=-1).gather(-1, targets[:, :, 0].unsqueeze(-1)).squeeze(-1)
                            lon_prob = F.softmax(lon_logits, dim=-1).gather(-1, targets[:, :, 1].unsqueeze(-1)).squeeze(-1)
                            sog_prob = F.softmax(sog_logits, dim=-1).gather(-1, targets[:, :, 2].unsqueeze(-1)).squeeze(-1)
                            cog_prob = F.softmax(cog_logits, dim=-1).gather(-1, targets[:, :, 3].unsqueeze(-1)).squeeze(-1)

                            # print(f"lat_logits:{lat_logits.shape}")
                            # print(f"lon_logits:{lon_logits.shape}")
                            # print(f"sog_logits:{sog_logits.shape}")
                            # print(f"cog_logits:{cog_logits.shape}")

                            lat_weight = lat_prob / total_lat_prob
                            lon_weight = lon_prob / total_lon_prob
                            sog_weight = sog_prob / total_sog_prob
                            cog_weight = cog_prob / total_cog_prob

                            # # 直接用logits计算
                            # lat_weight = torch.softmax(lat_logits, dim=2) / total_lat_prob
                            # lon_weight = torch.softmax(lon_logits, dim=2) / total_lon_prob
                            # sog_weight = torch.softmax(sog_logits, dim=2) / total_sog_prob
                            # cog_weight = torch.softmax(cog_logits, dim=2) / total_cog_prob

                            weight = (lat_weight.mean() + lon_weight.mean() + sog_weight.mean() + cog_weight.mean()) / 4
                            print(f"weight:{weight}")
                            # weight = 1
                            # print(f"weight:{weight}")

                            # 累积加权梯度
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    grad_accumulator[name] += weight * param.grad.clone()
                                    # if name == "blocks.blocks.modules_list.4.mlp.2.weight":
                                    #     print(f"  Accumulated gradient for {name}: {grad_accumulator[name]}")

                        # 将累积梯度应用到模型参数
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad = grad_accumulator[name]
                                # if name == "blocks.blocks.modules_list.4.mlp.2.weight":
                                #         print(f"  finally gradient for {name}: {grad_accumulator[name]}")

                        # 梯度裁剪和参数更新
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (
                                    seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                            if self.tokens < config.warmup_tokens:
                                # 线性预热策略
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # 余弦学习率衰减策略
                                progress = float(self.tokens - config.warmup_tokens) / float(
                                    max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # report progress 输出训练集结果
                        pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

                    # batch_mean = seqs.mean().item()
                    # batch_std = seqs.std().item()
                    # batch_means.append(batch_mean)
                    # batch_stds.append(batch_std)

            else:
                model.eval()
                for it, (seqs, target, masks, seqlens, mmsis, time_start) in pbar:
                    # place data on the correct device
                    seqs = seqs.to(self.device)
                    target = target.to(self.device)
                    masks = masks[:, :-1].to(self.device)

                    # forward the model
                    with torch.no_grad():
                        _, loss = model(seqs, target, masks=masks,
                                        weight_caculate=is_train, with_auxiliary=self.with_auxiliary)

                        loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())
                        d_loss += loss.item() * seqs.shape[0]
                        d_n += seqs.shape[0]

            if is_train:
                logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}.")
            else:
                logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")

            if not is_train:
                test_loss = float(np.mean(losses))
                return test_loss

            # plt.rcParams.update({'font.size': 17})
            #
            # plt.figure(figsize=(18, 6))
            #
            # # 绘制每个批次的均值和标准差的折线图
            # plt.subplot(1, 3, 1)
            # plt.plot(batch_means)
            # plt.xlabel("Batch")
            # plt.ylabel("Mean")
            # plt.title("Batch Means")
            # plt.legend()
            #
            # plt.subplot(1, 3, 2)
            # plt.plot(batch_stds)
            # plt.xlabel("Batch")
            # plt.ylabel("Standard Deviation")
            # plt.title("Batch Standard Deviations")
            # plt.legend()
            #
            # # 绘制每个批次的均值和标准差的散点图
            # plt.subplot(1, 3, 3)
            # plt.scatter(batch_means, batch_stds, label="Batch Stats")
            # plt.xlabel("Mean")
            # plt.ylabel("Standard Deviation")
            # plt.title("Batch Mean vs. Standard Deviation")
            # plt.legend()
            #
            # plot_file_path_combined = "batch_stats_combined.png"
            # plt.tight_layout()
            # plt.savefig(plot_file_path_combined)
            # plt.close()

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        # 配置Profiler
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_flops=True,
            profile_memory=True,
            with_stack=False
        )

        for epoch in range(config.max_epochs):

            # if epoch % 5 == 0:  # 每5个epoch记录一次
            #     with profiler as prof:
            #         run_epoch('Training', epoch=epoch)
            #     prof.export_chrome_trace(f"trace_epoch_{epoch}.json")
            # else:
            #     run_epoch('Training', epoch=epoch)
            # if self.test_dataset is not None:
            #     with record_function("model_validation_epoch"):
            #         test_loss = run_epoch('Valid', epoch=epoch)

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)

            ## SAMPLE AND PLOT
            # 验证集进行预测并绘制结果
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            seqs, target, masks, seqlens, mmsis, weight = next(iter(aisdls["test"]))
            seqs = seqs.to(self.device)
            target = target.to(self.device)
            n_plots = 7
            init_seqlen = INIT_SEQLEN  # 18
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds = sample(raw_model,
                           target,
                           seqs_init,
                           96 - init_seqlen,
                           temperature=1.0,
                           sample=True,
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k)

            if Config.valid_plot:
                img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')  # 定义图片的保存路径
                plt.figure(figsize=(9, 6), dpi=150)
                cmap = colormaps.get_cmap("jet")
                preds_np = preds.detach().cpu().numpy()
                inputs_np = seqs.detach().cpu().numpy()
                for idx in range(n_plots):
                    c = cmap(float(idx) / (n_plots))
                    try:
                        seqlen = seqlens[idx].item()
                    except:
                        continue
                    plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
                    plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3,
                             color=c)  # 绘制起始的部分路径
                    plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.",
                             color=c)  # 绘制全部真实路径
                    plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4,
                             color=c)  # 绘制预测点位
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(img_path, dpi=150)
                plt.close()

        # Final state保存最后一次训练结果
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        logging.info(f"Last epoch: {epoch + 1:03d}, saving model to model_{epoch + 1:03d}.pt")

        torch.save(raw_model.state_dict(), save_path)
