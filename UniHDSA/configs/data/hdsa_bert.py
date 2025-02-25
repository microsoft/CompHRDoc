from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from projects.unified_layout_analysis_v2.evaluation.unified_layout_evaluation import UniLayoutEvaluator
from projects.unified_layout_analysis_v2.modeling.backbone.bert import TextTokenizer

from detrex.data.dataset_mappers import PODDatasetMapper, pod_transform_gen
from detrex.data.dataset_mappers import HRDocDatasetMapper

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="COMP_HRDOC_HR_TRAIN"),
    mapper=L(HRDocDatasetMapper)(
        augmentation=L(pod_transform_gen)(
            min_size_train=(320, 416, 512, 608, 704, 800),
            max_size_train=1024,
            min_size_train_sampling="choice",
            min_size_test=512,
            max_size_test=1024,
            random_resize_type="ResizeShortestEdge",
            random_flip=False,
            is_train=True,
        ),
        TextTokenizer=L(TextTokenizer)(
            model_type="bert-base-uncased",
            text_max_len=512,
            input_overlap_stride=0,
        ),
        is_train=True,
        image_format="BGR",
    ),
    total_batch_size=1,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="COMP_HRDOC_HR_TEST"),
    mapper=L(HRDocDatasetMapper)(
        augmentation=L(pod_transform_gen)(
            min_size_train=(320, 416, 512, 608, 704, 800),
            max_size_train=1024,
            min_size_train_sampling="choice",
            min_size_test=512,
            max_size_test=1024,
            random_resize_type="ResizeShortestEdge",
            random_flip=False,
            is_train=False,
        ),
        TextTokenizer=L(TextTokenizer)(
            model_type="bert-base-uncased",
            text_max_len=512,
            input_overlap_stride=0,
        ),
        is_train=False,
        image_format="BGR",
    ),
    num_workers=4,
)

dataloader.evaluator = L(UniLayoutEvaluator)(
    dataset_name="${..test.dataset.names}",
)