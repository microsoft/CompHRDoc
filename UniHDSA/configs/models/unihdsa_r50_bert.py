import torch.nn as nn
import copy
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import ResNet, BasicStem
from detrex.modeling.neck import ChannelMapper
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L
from detrex.modeling.matcher import HungarianMatcher

from projects.unified_layout_analysis_v2.modeling import (
    UniDETRMultiScales,
    DabDeformableDetrTransformer,
    DabDeformableDetrTransformerEncoder,
    DabDeformableDetrTransformerDecoder,
    TwoStageCriterion
)

from projects.unified_layout_analysis_v2.modeling.uni_relation_prediction_head import (
    UniRelationPredictionHead,
    HRIPNHead
)

from projects.unified_layout_analysis_v2.modeling.doc_transformer import (
    DocTransformerEncoder,
    DocTransformer
)

from projects.unified_layout_analysis_v2.modeling.backbone.bert import (
    Bert,
    TextTokenizer
)

# Define the main model
model = L(UniDETRMultiScales)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
        freeze_at=1,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "res3": ShapeSpec(channels=512),
            "res4": ShapeSpec(channels=1024),
            "res5": ShapeSpec(channels=2048),
        },
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DabDeformableDetrTransformer)(
        encoder=L(DabDeformableDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=3,
            post_norm=False,
            num_feature_levels=4,
        ),
        decoder=L(DabDeformableDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=3,
            return_intermediate=True,
            num_feature_levels=4,
        ),
        as_two_stage=True,
        num_feature_levels=4,
        decoder_in_feature_level=[0, 1, 2, 3],
    ),
    embed_dim=256,
    num_classes=14,
    num_graphical_classes=2,
    num_types=3,
    relation_prediction_head=L(UniRelationPredictionHead)(
        relation_num_classes=2,
        embed_dim=256,
        hidden_dim=1024,
    ), # 0: a->a, 1: intra, 2: inter
    aux_loss=True,
    criterion=L(TwoStageCriterion)(
        num_classes=2,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    as_two_stage=True,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
    windows_size=[6,8],
    freeze_language_model=False,
)

model.logical_role_relation_prediction_head=L(UniRelationPredictionHead)(
    relation_num_classes=1,
    embed_dim=256,
    hidden_dim=1024,
)

# Update auxiliary loss weight dictionary
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {f"{k}_{i}": v for i in range(model.transformer.decoder.num_layers - 1) for k, v in base_weight_dict.items()}
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

# Additional loss weight updates
model.criterion.weight_dict.update({
    "loss_class_enc": 1.0,
    "loss_bbox_enc": 5.0,
    "loss_giou_enc": 2.0,
})

# Add document transformer module
model.doc_transformer = L(DocTransformer)(
    encoder=L(DocTransformerEncoder)(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        num_layers=3,
        post_norm=False,
        batch_first=True,
    ),
    decoder=None,
)

# Add relation prediction head
model.doc_relation_prediction_head = L(HRIPNHead)(
    relation_num_classes=2,
    embed_dim=256,
    hidden_dim=1024,
)

# Add language model
model.language_model = L(Bert)(
    bert_model_type="bert-base-uncased",
    text_max_len=512,
    input_overlap_stride=0,
    output_embedding_dim=1024,
    max_batch_size=1,
    used_layers=12,
    used_hidden_idxs=[12],
    hidden_embedding_dim=768,
)

# Add tokenizer
model.tokenizer = L(TextTokenizer)(
    model_type="bert-base-uncased",
    text_max_len=512,
    input_overlap_stride=0,
)
