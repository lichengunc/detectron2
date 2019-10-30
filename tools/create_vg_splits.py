"""
Revised from https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/create_splits.py
"""
import argparse
import os
import os.path as osp
import random
from random import shuffle
import shutil
import subprocess
import sys
import json

random.seed(10) # Make dataset splits repeatable

CURDIR = osp.dirname(osp.realpath(__file__))

# The root directory which holds all information of the dataset.
splitDir = 'datasets/coco/karpathy_splits'
dataDir = 'datasets/vg'

train_list_file = osp.join(splitDir, 'train.txt')
val_list_file = osp.join(splitDir, 'val.txt')
test_list_file = osp.join(splitDir, 'test.txt')

# Forbidden coco_image_ids and vg_image_ids, including 
# refer_val_coco_iids, refer_test_coco_iids
# flickr30k_coco_iids, flickr30k_vg_iids
excluded_coco_vg_iids = json.load(
        open('datasets/prepro/excluded_coco_vg_iids.json', 'r'))

# First determine train, val, test splits (x, 5000, 5000)
train = set()
val = set()
test = set()

# Load karpathy coco splits
karpathy_train = set()
with open(osp.join(splitDir, 'karpathy_train_images.txt')) as f:
    for line in f.readlines():
        image_id = int(line.split('.')[0].split('_')[-1])
        karpathy_train.add(image_id)

karpathy_val = set()
with open(osp.join(splitDir, 'karpathy_val_images.txt')) as f:
    for line in f.readlines():
        image_id=int(line.split('.')[0].split('_')[-1])
        karpathy_val.add(image_id)
    
karpathy_test = set()
with open(osp.join(splitDir, 'karpathy_test_images.txt')) as f:
    for line in f.readlines():
        image_id=int(line.split('.')[0].split('_')[-1])
        karpathy_test.add(image_id)
print(f"Karpathy splits are {len(karpathy_train)}, {len(karpathy_val)}, "
      f"{len(karpathy_test)} (train, val, test)")

# Load VG image metadata
coco_ids = set()
coco_id_to_vg_id = {}
with open(osp.join(dataDir, 'image_data.json')) as f:
    metadata = json.load(f)
    for item in metadata:
        if item['coco_id']:
            coco_ids.add(item['coco_id'])
            coco_id_to_vg_id[item['coco_id']] = item['image_id']
print(f"Found {len(coco_ids)} visual genome images claiming coco ids")
print(f"Overlap with Karpathy train is {len(karpathy_train & coco_ids)}")
print(f"Overlap with Karpathy val is {len(karpathy_val & coco_ids)}")
print(f"Overlap with Karpathy test is {len(karpathy_test & coco_ids)}")    

# Determine splits
remainder = []
for item in metadata:
    if item['coco_id']:
        if item['coco_id'] in karpathy_train:
            train.add(item['image_id'])
        elif item['coco_id'] in karpathy_val:
            val.add(item['image_id'])
        elif item['coco_id'] in karpathy_test:
            test.add(item['image_id'])    
        else:
            remainder.append(item['image_id'])
    else:
        remainder.append(item['image_id'])
shuffle(remainder)
while len(test) < 5000:
    test.add(remainder.pop())
while len(val) < 5000:
    val.add(remainder.pop())
train |= set(remainder)

assert len(test) == 5000
assert len(val) == 5000
assert len(train) == len(metadata) - 10000

# remove some excluded image_ids from train and val
excluded_image_ids = set()
for coco_id in excluded_coco_vg_iids['refer_val_coco_iids'] + \
               excluded_coco_vg_iids['refer_test_coco_iids'] + \
               excluded_coco_vg_iids['flickr30k_coco_iids']:
    if coco_id in coco_id_to_vg_id:
        excluded_image_ids.add(coco_id_to_vg_id[coco_id])
excluded_image_ids = excluded_image_ids.union(
                        set(excluded_coco_vg_iids['flickr30k_vg_iids']))
print(f'{len(excluded_image_ids)} images will be excluded.')

train = train.difference(excluded_image_ids)
val = val.difference(excluded_image_ids)
test = test.difference(excluded_image_ids)
assert len(train.intersection(val)) == 0
assert len(train.intersection(test)) == 0
assert len(val.intersection(test)) == 0
print(f'After removing excluded_coco_vg_iids, we have {len(train)}, '
      f'{len(val)}, {len(test)} (train, val, test).')

# create splits
image_data = {'train': [], 'val': [], 'test': []}
for split_name, split in zip(['train', 'val', 'test'], [train, val, test]):
    for item in metadata:
        # must-have: file_name, height, width, id, 
        if 'image_id' in item and item['image_id'] in split:
            url = item['url'].split('/')
            file_name = f'{url[-2]}/{url[-1]}'
            item['file_name'] = file_name
            item['id'] = item['image_id']
            del item['url']
            del item['image_id']
            image_data[split_name].append(item)

with open('datasets/prepro/vg_image_data.json', 'w') as f:
    json.dump(image_data, f)
print('datasets/prepro/vg_image_data.json prepared.')