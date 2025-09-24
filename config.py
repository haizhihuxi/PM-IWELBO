

import os
import torch


class Config():
    retrain = True
    auxiliary_learning = False
    half_train = False
    similarity_retrain = False
    sim_test = False
    sim_train_plot_result = False
    tb_log = False
    plot_result = True
    valid_plot = True
    device = torch.device("cuda:0")

    max_epochs = 20
    max_sim_epochs = 10
    batch_size = 32
    n_samples = 16

    init_seqlen = 18
    max_seqlen = 140
    min_seqlen = 36

    input_dim = 2
    hidden_dim = 256
    output_dim = 10
    window_size_judge = 10
    window_size_info = 5

    dataset_name = "ct_dma" #“american”

    lat_size = 250
    lon_size = 270
    lat_target_size = 250
    lon_target_size = 270
    sog_size = 32
    cog_size = 72

    n_lat_embd = 256
    n_lon_embd = 256
    n_lat_target_embd = 256
    n_lon_target_embd = 256
    n_sog_embd = 128
    n_cog_embd = 128

    lat_min = 55.5
    lat_max = 58.0
    lon_min = 10.3
    lon_max = 13

    # ===========================================================================
    # Model and sampling flags
    mode = "pos"
    sample_mode = "pos_vicinity"
    top_k = 10  # int or None
    r_vicinity = 40  # int
    frechet_judge = False
    sspd_judge = True


    show_parameter = False

    # Data flags
    # ===================================================
    datadir = f"./data/{dataset_name}/"  # ./data/ct_dma/
    if dataset_name == "ct_dma":
        trainset_name = f"{dataset_name}_train_erp_dtw.pkl"  # ct_dma_train.pkl
        validset_name = f"{dataset_name}_valid_剔除无效数据.pkl"
        testset_name = f"{dataset_name}_test_剔除无效数据.pkl"
    if dataset_name == "american":
        trainset_name = f"train.pkl"
        validset_name = f"valid.pkl"
        testset_name = f"test.pkl"

    # model parameters
    # ===================================================
    n_head = 8
    n_layer = 8
    n_adaptive_layer = 1
    traj_compary_threshold = 0.05
    threathod_10 = 0.2
    threathod_5 = 0.2
    weight_10 = 0.5
    full_size = lat_size + lon_size + sog_size + cog_size
    full_des_size = lat_target_size + lon_target_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd  # 768
    n_auxil = n_lat_target_embd + n_lon_target_embd
    # base GPT config, params common to all GPT versions
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    positive_value = 0.1  # 增加的权重值
    start = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    end = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd + n_lat_embd + n_lon_embd

    # optimization parameters
    # ===================================================
    learning_rate = 9e-5  # 6e-4 0.0006 9e-5
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    grad_norm_clip_sim = 0.5
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original 模型在训练初期更快地收敛，然后在训练后期进行更精细的参数调整。
    lr_decay = True
    warmup_tokens = 512 * 20  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    num_workers = 12  # for DataLoader


    filename = "result"
    savedir = "./new-data-results/" + filename + "/"
    savedir_result_picture = "./picture-results/" + filename + "/result picture real/"
    savedir_test = "./results/" + filename + "/"
    sim_outfile = 'evaluation_results.txt'

    ckpt_path = os.path.join(savedir, "model.pt")
    ckpt_path_sim = os.path.join(savedir, "model_sim.pt")
