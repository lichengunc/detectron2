# -*- coding: utf-8 -*-
# Copyright (c) Microsoft, Inc. and its affiliates. All Rights Reserved

import contextlib
import logging
import numpy as np
import io
import os
import itertools
from tqdm import tqdm
import tempfile
import torch
from collections import OrderedDict, defaultdict
from fvcore.common.file_io import PathManager
from functools import lru_cache

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.genome import Genome

from .evaluator import DatasetEvaluator

import sys
import pdb


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class AttributeEvaluator(DatasetEvaluator):
    """
    Evaluate attributes
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                    so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """ 
        assert cfg.TEST.EVAL_ATTRIBUTE is True
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._is_2007 = False  # we do not used voc_2007 AP metric on default
        self._meta_data = MetadataCatalog.get(dataset_name)
        self._class_names = self._meta_data.attribute_classes
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        json_file = PathManager.get_local_path(self._meta_data.json_file)
        self._genome_api = Genome(json_file)

    def reset(self):
        self._predictions = defaultdict(list)
    
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            gt_boxes = instances.gt_boxes.tensor.numpy()
            gt_attributes = instances.gt_attributes.numpy()  # (num_objs, num_attrs)
            assert len(gt_boxes) == len(boxes)
            scores, classes = instances.pred_attr_probs.max(1)
            scores = scores.tolist()
            classes = classes.tolist()
            for box, gt_box, score, cls, gt_attrs in zip(boxes, gt_boxes, scores, classes, gt_attributes):
                if gt_attrs.sum() > 0:
                    # we only consider gt_boxes with attributes
                    xmin, ymin, xmax, ymax = box
                    self._predictions[cls].append(
                        f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                    )

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(list(itertools.chain(*all_predictions)), f)

        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} attributes using voc_{} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="attribute_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = []  # ap per class
            nposs = []  # npos per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                thresh = 50
                rec, prec, ap, npos = attribute_eval(
                    res_file_template,
                    self._genome_api,
                    self._meta_data,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007
                )
                aps += [ap * 100]
                nposs += [npos]
                print('\rAP for {}.{} = {:.4f} (npos={:,})'.format(cls_id, cls_name, ap * 100, npos), 
                      end='       ')

        ret = OrderedDict()
        ap50 = np.mean(aps)
        weights = np.array(nposs)
        weights = weights / weights.sum()
        weighted_ap50 = np.average(aps, weights=weights)
        ret["bbox"] = {"AP50": ap50, "weighted_AP50": weighted_ap50}
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

def parse_rec(anno_dict_list, genome_api):
    objects = []
    for anno in anno_dict_list:
        obj = {}
        obj["names"] = [genome_api.atts[att_id]["name"] for att_id in anno["attribute_ids"]]
        obj["difficult"] = 0
        obj["bbox"] = [anno["bbox"][0],
                       anno["bbox"][1],
                       anno["bbox"][0] + anno["bbox"][2] - 1,
                       anno["bbox"][1] + anno["bbox"][3] - 1]
        objects.append(obj)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def attribute_eval(detpath, genome_api, meta, classname, ovthresh=0.5,
                   use_07_metric=False):
    # read list of images
    img_ids = sorted(list(genome_api.imgs.keys()))
    imagenames = [str(img_id) for img_id in img_ids]
    imgs = genome_api.loadImgs(img_ids)
    anns = [genome_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))

    # load annots
    recs = {}
    for (img_dict, anno_dict_list) in imgs_anns:
        imagename = str(img_dict["id"])
        recs[imagename] = parse_rec(anno_dict_list, genome_api)
    
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if classname in obj["names"]]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if npos == 0:
        # No ground truth examples
        print("no ground-truth")
        return 0, 0, 0, 0

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, npos
