from alfred.utils.log import logger as logging
import os
import torch

from configs.ct_coco_r50_config import config

from models.data import MetadataCatalog
from models.evaluation.coco_evaluation import COCOEvaluator
from models.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator

from typing import Any, Dict, List
import argparse

from models.centernet import build_model
from models.train.trainer import DefaultTrainer
from models.evaluation.evaluator import DatasetEvaluators
from models.train import hooks
from alfred.utils.log import logger
from alfred.dl.torch.common import device
from models.structures.boxes import Boxes
from models.structures.instances import Instances

def default_argument_parser():
    parser = argparse.ArgumentParser(description="CenterNet Pro Train")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format('9080'))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    config.merge_from_list(args.opts)
    cfg = config
    model = build_model(cfg)
    logger.info('model build.')

    a = torch.rand([3, 512, 512]).to(device)
    boxes = Boxes(torch.rand([7, 4]))
    classes = torch.tensor([3, 5, 6, 6, 8, 12, 23])
    instances = Instances(image_size=[(679, 345)])
    instances.gt_boxes = boxes
    instances.gt_classes = classes
    x = [{'image': a, 'instances': instances}]
    b = model(x)
    print(b)
    for k, v in b.items():
        print('{}: {}'.format(k, v.shape))
