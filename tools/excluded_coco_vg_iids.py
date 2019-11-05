"""
We will exclude the following coco image ids for pretraining:
1) karpathy's minival split
2) refcoco/refcoco+/refcocog's val+test split
3) duplicated Flickr30k images in COCO
Note, karpathy's minival split will be our val split for pretraining, but 
refcoco*'s test split will only be used for testing.
"""
import os
import os.path as osp
import json
import pickle

# paths
this_dir = osp.dirname(__file__)
data_dir = osp.join(this_dir, '..', 'datasets')
vg_dir = osp.join(data_dir, 'vg')
coco_dir = osp.join(data_dir, 'coco')
refer_dir = osp.join(data_dir, 'refer')
flickr_dir = osp.join(data_dir, 'flickr30k')
genome_dir = osp.join(data_dir, 'genome')

# Excluded refcoco*'s val+test images
refcoco_data = pickle.load(
    open(osp.join(refer_dir, 'refcoco/refs(unc).p'), 'rb'))
refcocog_data = pickle.load(
    open(osp.join(refer_dir, 'refcocog/refs(umd).p'), 'rb'))
refer_val_coco_iids = []
refer_test_coco_iids = []
for ref in refcoco_data:
    if ref['split'] in ['testA', 'testB']:
        refer_test_coco_iids.append(ref['image_id'])
    if ref['split'] == 'val':
        refer_val_coco_iids.append(ref['image_id'])
for ref in refcocog_data:
    if ref['split'] in ['test']:
        refer_test_coco_iids.append(ref['image_id'])
    if ref['split'] == 'val':
        refer_val_coco_iids.append(ref['image_id'])
refer_val_coco_iids_set = set(refer_val_coco_iids)
refer_test_coco_iids_set = set(refer_test_coco_iids)
print(f'- There are {len(refer_val_coco_iids_set)} refcoco_unc + refcocog_umd '
      f'[val]  coco images.')
print(f'- There are {len(refer_test_coco_iids_set)} refcoco_unc + refcocog_umd '
      f'[test] coco images.')

# Load Karpathy's minival (a subset of COCO's val)
karpathy_train_iids = []
karpathy_train_file = open(
    osp.join(genome_dir, 'image_splits', 'karpathy_train_images.txt'), 'r')
for x in karpathy_train_file.readlines():
    karpathy_train_iids.append(int(x.split()[1]))
karpathy_train_set = set(karpathy_train_iids)
assert len(karpathy_train_set) == len(karpathy_train_iids)
print('COCO\'s [karpathy_train] %s images loaded.' % len(karpathy_train_set))

karpathy_val_iids = []
karpathy_val_file = open(
    osp.join(genome_dir, 'image_splits', 'karpathy_val_images.txt'), 'r')
f = karpathy_val_file.readlines()
for x in f:
    karpathy_val_iids.append(int(x.split()[1]))
karpathy_val_set = set(karpathy_val_iids)
assert len(karpathy_val_set) == len(karpathy_val_iids)
print('COCO\'s [karpathy_val] %s images loaded.' % len(karpathy_val_set))

karpathy_test_iids = []
karpathy_test_file = open(
    osp.join(genome_dir, 'image_splits', 'karpathy_test_images.txt'), 'r')
f = karpathy_test_file.readlines()
for x in f:
    karpathy_test_iids.append(int(x.split()[1]))
karpathy_test_set = set(karpathy_test_iids)
assert len(karpathy_test_set) == len(karpathy_test_iids)
print('COCO\'s [karpathy_test] %s images loaded.' % len(karpathy_test_set))

# Excluding all Flickr30K images
flickr30k_coco_iids = []
flickr30k_vg_iids = []
flickr30k_url_ids = set()
for url_id in open(
            osp.join(flickr_dir, 'flickr30k_entities', 'train.txt'), 'r'
        ).readlines():
    flickr30k_url_ids.add(int(url_id))
for url_id in open(
            osp.join(flickr_dir, 'flickr30k_entities', 'val.txt'), 'r'
        ).readlines():
    flickr30k_url_ids.add(int(url_id))
for url_id in open(
            osp.join(flickr_dir, 'flickr30k_entities', 'test.txt'), 'r'
        ).readlines():
    flickr30k_url_ids.add(int(url_id))
print('There are %s flickr30k_url_ids.' % len(flickr30k_url_ids))

coco_image_data = json.load(open(
        osp.join(coco_dir, 'annotations', 'instances_train2014.json'))
    )['images'] + \
    json.load(open(
        osp.join(coco_dir, 'annotations', 'instances_val2014.json'))
    )['images']
for img in coco_image_data:
    # example: 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg'
    url_id = int(img['flickr_url'].split('/')[-1].split('_')[0])
    if url_id in flickr30k_url_ids:
        flickr30k_coco_iids.append(img['id'])
print('%s coco images were found in Flickr30K.' % len(flickr30k_coco_iids))

vg_image_data = json.load(open(osp.join(vg_dir, 'image_data.json')))
for img in vg_image_data:
    if img['flickr_id'] is not None:
        url_id = int(img['flickr_id'])
        if url_id in flickr30k_url_ids:
            flickr30k_vg_iids.append(img['image_id'])
print('%s vg images were found in Flickr30K.' % len(flickr30k_vg_iids))

# # test
# print(len(refer_val_coco_iids_set.intersection(refer_test_coco_iids_set)))
# print(len(karpathy_val_set.intersection(karpathy_test_set)))

# Save
output = {'refer_val_coco_iids': list(refer_val_coco_iids_set), 
          'refer_test_coco_iids': list(refer_test_coco_iids_set), 
          'flickr30k_coco_iids': flickr30k_coco_iids, 
          'flickr30k_vg_iids': flickr30k_vg_iids,
          'karpathy_train_iids': list(karpathy_train_iids),
          'karpathy_val_iids': list(karpathy_val_iids), 
          'karpathy_test_iids': list(karpathy_test_iids)}
if not osp.isdir('datasets/genome'): os.makedirs('datasets/genome')
with open('datasets/genome/excluded_coco_vg_iids.json', 'w') as f:
    json.dump(output, f)
print('datasets/genome/excluded_coco_vg_iids.json saved.')
