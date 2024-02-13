g_dino_ckpt_path = "{path_to_GroundingDINO_ckpt}" # https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

checkpoint = None # path to your checkpoint, set None if train from scratch
model_path = "flan-t5-base"

stage = "stage1" # stage1 for pre-training and stage2 for fine-tuning
device = "cuda:0"
save_path = "fold_to_save_checkpoint"
exp_name = "cofi_para"
model_type = 't5'

record_dir = "./logs"
res_dir = "./outputs/"
skip_training = False
seed = 42
batch_size = 4
epochs = 10
num_train = -1
num_val = -1
num_workers = 4
lr = 5e-5
adam_eps = 1e-8
warmup_steps = 0
max_grad_norm =  1.0
use_wandb = False
alpha = 0.8

max_src_len = 512
max_tgt_len = 200
num_heads = 1
