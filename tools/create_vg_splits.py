"""
The original splits are from 
https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/create_splits.py

Tried to reproduce Peter's [train,val,test] image_ids but failed (may due to random seed), so 
start from Peter's splits directly.
"""
import argparse
import os
import os.path as osp
import sys
import json

def main(args):
    # The root directory which holds all information of the dataset.
    split_dir = args.split_dir
    data_dir = args.data_dir

    train_list_file = osp.join(split_dir, 'train.txt')
    val_list_file = osp.join(split_dir, 'val.txt')
    test_list_file = osp.join(split_dir, 'test.txt')

    # First determine train, val, test splits (x, 5000, 5000)
    train = set()
    val = set()
    test = set()

    train = set()
    with open(osp.join(split_dir, 'train.txt')) as f:
        for line in f.readlines():
            image_id = int(line.split(' ')[0].split('/')[1].split('.')[0])
            train.add(image_id)
    
    val = set()
    with open(osp.join(split_dir, 'val.txt')) as f:
        for line in f.readlines():
            image_id = int(line.split(' ')[0].split('/')[1].split('.')[0])
            val.add(image_id) 

    test = set()
    with open(osp.join(split_dir, 'test.txt')) as f:
        for line in f.readlines():
            image_id = int(line.split(' ')[0].split('/')[1].split('.')[0])
            test.add(image_id)

    # Load VG image metadata
    coco_ids = set()
    coco_id_to_vg_id = {}
    with open(osp.join(data_dir, 'image_data.json')) as f:
        metadata = json.load(f)
        for item in metadata:
            if item['coco_id']:
                coco_ids.add(item['coco_id'])
                coco_id_to_vg_id[item['coco_id']] = item['image_id']
    print(f"Found {len(coco_ids)} visual genome images claiming coco ids")

    # Determine split
    assert len(test) == 5000
    assert len(val) == 5000
    assert len(train) == len(metadata) - 10000
    print(f'We have {len(train)}, {len(val)}, {len(test)} (train, val, test).')

    # remove some excluded image_ids from train and val
    if args.filter:
        # Forbidden coco_image_ids and vg_image_ids, including 
        # refer_val_coco_iids, refer_test_coco_iids
        # flickr30k_coco_iids, flickr30k_vg_iids
        excluded_coco_vg_iids = json.load(
                open(args.excluded_iids_json, 'r'))
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

    # save image_data.json or filtered_image_data.json
    file_name = 'image_data.json' if not args.filter \
                else 'filtered_image_data.json'
    output_file = osp.join(args.genome_dir, file_name)
    with open(output_file, 'w') as f:
        json.dump(image_data, f)
    print(f'{output_file} prepared.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', type=str, 
                        default='datasets/genome/image_splits')
    parser.add_argument('--data_dir', type=str, 
                        default='datasets/vg')
    parser.add_argument('--genome_dir', type=str, 
                        default='datasets/genome')
    parser.add_argument('--filter', dest='filter', 
                        action='store_true')
    parser.add_argument('--excluded_iids_json', dest='excluded_iids_json', 
                        default='datasets/genome/excluded_coco_vg_iids.json')
    args = parser.parse_args()

    main(args)