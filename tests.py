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


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(
                COCOEvaluator(
                    dataset_name, cfg, True,
                    output_folder, dump=cfg.GLOBAL.DUMP_TRAIN
                ))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def train(args):
    config.merge_from_list(args.opts)
    cfg = config
    model = build_model(cfg)
    logger.info(f"Model structure: {model}")
    trainer = Trainer(cfg, model)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


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
    train(args)
