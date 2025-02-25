from .data.hdsa_bert import dataloader
from .models.unihdsa_r50_bert import model
from detrex.config import get_config
from detectron2.config import LazyCall as L
from ..modeling.uni_relation_prediction_head import UniRelationPredictionHead

optimizer = get_config("common/optim.py").AdamW
# modify optimizer config
optimizer.lr = 4e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-2
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else (0.2 if "bert" in module_name else 1)

# modify training config
train = get_config("common/train.py").train

# initialize checkpoint to be loaded

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 1000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"

train.ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=True,
        fp16_compression=False,
)

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/"
dataloader.evaluator.output_dir = train.output_dir
dataloader.evaluator.as_two_stage = True
train.wandb = dict(
    enabled=True,
    params=dict(
        dir="./wandb_output",
        project="",
        name="",
    )
)

dataloader.train.total_batch_size = 16 # one GPU one Document # 2Nodes

# max training iterations
train.max_iter = 22500

# run evaluation every 6000 iters
train.eval_period = 6000

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

# model.freeze_language_model = True
# model.language_model.used_hidden_idxs = [6]
# model.language_model.used_layers = 6