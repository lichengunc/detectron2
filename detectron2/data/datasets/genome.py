import os
import os.path as osp
import json
import logging
import collections
import contextlib

from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)

__all__ = ["load_genome_instances", "register_genome_instances"]


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Genome():
    def __init__(self, json_file):
        # dataset = {images, annotations, attributes, categories}
        dataset = json.load(open(json_file, 'r'))
        self.dataset = dataset 
        self.createIndex()
    
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, atts, imgs = {}, {}, {}, {}
        imgToAnns = collections.defaultdict(list)
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann
        for img in self.dataset['images']:
            imgs[img['id']] = img
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat
        for att in self.dataset['attributes']:
            atts[att['id']] = att

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        self.cats = cats
        self.atts = atts
    
    def getAttIds(self):
        atts = self.dataset['attributes']
        ids = [att['id'] for att in atts]
        return ids
    
    def getCatIds(self):
        cats = self.dataset['categories']
        ids = [cat['id'] for cat in cats]
        return ids

    def loadImgs(self, ids=[]):
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def loadCats(self, ids=[]):
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadAtts(self, ids=[]):
        if _isArrayLike(ids):
            return [self.atts[id] for id in ids]
        elif type(ids) == int:
            return [self.atts[ids]]

def load_genome_instances(json_file, image_root, dataset_name):
    """
    Load Genome detection annotations to Detectron2 format
    """
    timer = Timer()
    genome_api = Genome(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    meta = MetadataCatalog.get(dataset_name)

    # object categories
    cat_ids = sorted(genome_api.getCatIds())
    cats = genome_api.loadCats(cat_ids)
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes
    meta.thing_dataset_id_to_contiguous_id = {v: i for i, v in enumerate(cat_ids)}

    # attribute categories
    att_ids = sorted(genome_api.getAttIds())
    atts = genome_api.loadAtts(att_ids)
    attribute_classes = [a["name"] for a in sorted(atts, key=lambda x: x["id"])]
    meta.attribute_classes = attribute_classes
    meta.attribute_dataset_id_to_contiguous_id = {v: i for i, v in enumerate(att_ids)}

    # sort indices for reproducible results
    img_ids = sorted(list(genome_api.imgs.keys()))
    imgs = genome_api.loadImgs(img_ids)
    anns = [genome_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    ann_keys = ["bbox", "category_id", "attribute_ids"]
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = osp.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            obj = {key: anno[key] for key in ann_keys if key in anno}
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = meta.thing_dataset_id_to_contiguous_id[obj["category_id"]]
            obj["attribute_ids"] = [meta.attribute_dataset_id_to_contiguous_id[att_id] 
                                    for att_id in obj["attribute_ids"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def register_genome_instances(name, json_file, image_root, meta_file):
    DatasetCatalog.register(name, lambda: load_genome_instances(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="genome"
    )
    # In case DatasetCatalog is not called, we manually add additional meta info
    meta = MetadataCatalog.get(name)
    meta_info = json.load(open(meta_file, "r"))
    cat_ids = sorted([c["id"] for c in meta_info["categories"]])
    meta.thing_classes = [c["name"] for c in sorted(meta_info["categories"], 
                                                    key=lambda x: x["id"])]
    meta.thing_dataset_id_to_contiguous_id = {v: i for i, v in enumerate(cat_ids)}
    att_ids = sorted([a["id"] for a in meta_info["attributes"]])
    meta.attribute_classes = [a["name"] for a in sorted(meta_info["attributes"], 
                                                        key=lambda x: x["id"])]
    meta.attribute_dataset_id_to_contiguous_id = {v: i for i, v in enumerate(att_ids)}
    # attribute count, stats from training portion of genome data 
    att_to_cnt = meta_info['att_to_cnt']
    meta.attribute_cnts = [att_to_cnt[a["name"]] for a in sorted(meta_info["attributes"], 
                                                                 key=lambda x: x["id"])]

    