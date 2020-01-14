import argparse
import os
import os.path as osp
import json
from fvcore.common.file_io import PathManager
import time

import torch
import torch.utils.data

from detectron2.structures import BoxMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch

from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger

from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import samplers
from detectron2.data.build import trivial_batch_collator

from detectron2.evaluation import extract_feats


def default_setup(cfg, args):
    output_dir = args.output_dir 
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file"):
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.model_weights
    default_setup(cfg, args)
    # In order to bypass dataset_mapper's pop()
    # ugly...TODO
    cfg.TEST.EVAL_ATTRIBUTE = True  
    cfg.freeze()
    return cfg

def get_dataset_dicts(args):
    """
    args:
    - image_root : containing all images
    - json_file  : [file_name, bboxes] where bboxes = [bbox (xywh)]
    """
    assert osp.isdir(args.image_root), f"{args.image_root} does not exist."
    imgs = json.load(open(args.json_file, 'r'))
    dataset_dicts = []
    for img_dict in imgs:
        # if osp.exists(osp.join(args.output_dir, 'features', img_dict["file_name"]+'.npz')):
        #     # TODO: this strategy is in case some process was dead, then
        #     # we just re-run this code again by avoiding extracting features
        #     # on existing image files.
        #     continue
        record = {}
        record["image_name"] = img_dict["file_name"]
        record["file_name"] = osp.join(args.image_root, img_dict["file_name"])  # absolute
        objs = []
        for bbox in img_dict["bboxes"]:
            obj = {"bbox": bbox,  # xywh
                   "bbox_mode": BoxMode.XYWH_ABS,
                   "category_id": 0  # fake some category_id, no use
                   }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def build_detection_loader(cfg, args, mapper=None):
    dataset_dicts = get_dataset_dicts(args)
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, is_train=False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    data_loader = build_detection_loader(cfg, args)
    extract_feats(model, data_loader, osp.join(args.output_dir, 'features'), given_boxes=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detectrion2 Feature Extraction")
    parser.add_argument("--json-file", type=str,
                        default="tmp/val2017_examples.json",
                        help="input image folder")
    parser.add_argument("--image-root", type=str, 
                        default="datasets/coco/val2017", 
                        help="input image folder")
    parser.add_argument("--output-dir", type=str, 
                        default="output/extracted_features", 
                        help="output folder")
    parser.add_argument("--config-file", metavar="FILE", 
                        default="configs/GENOME-Detection/faster_rcnn_softmax_attr_R_101_C4_3x.yaml", 
                        help="path to config file")
    parser.add_argument("--model-weights", type=str, 
                        default="output/genome_faster_rcnn_softmax0.5_expminus_attr_R_101_C4_3x/model_final.pth",
                        help="path to final model")
    parser.add_argument("--num-gpus", type=int, 
                        default=8, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, 
                        default=1)
    parser.add_argument("--machine-rank", type=int, 
                        default=0, 
                        help="the rank of this machine (unique per machine)"
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    args = parser.parse_args()
    print("Command Line Args:", args)

    # run
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )