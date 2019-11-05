import os
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import json
import argparse 
import time

class vg():
    def __init__(self, genome_dir, vocab_dir, image_set, filter):

        # image_data = {split: [{id, file_name, coco_id, flickr_id, height, width}]}
        image_data_path = osp.join(genome_dir, 'image_data.json') if not filter \
                          else osp.join(genome_dir, 'filtered_image_data.json')
        self.image_data = json.load(open(image_data_path, 'r'))
        self.iid_to_meta = {img['id']: img for img in self.image_data[image_set]}

        # other directories
        self.vocab_dir = vocab_dir
        self.genome_dir = genome_dir
        self._image_set = image_set
 
        # Load classes
        self._classes = []
        self._class_to_ind = {}  # start from 1
        with open(osp.join(self.vocab_dir, 'objects_vocab.txt')) as f:
            count = 1
            for object in f.readlines():
                names = [n.lower().strip() for n in object.split(',')]
                self._classes.append(names[0])
                for n in names:
                    self._class_to_ind[n] = count
                count += 1 

        # Load attributes
        self._attributes = []
        self._attribute_to_ind = {}  # start from 1
        with open(osp.join(self.vocab_dir, 'attributes_vocab.txt')) as f:
            count = 1
            for att in f.readlines():
                names = [n.lower().strip() for n in att.split(',')]
                self._attributes.append(names[0])
                for n in names:
                    self._attribute_to_ind[n] = count
                count += 1           

        # Load relations
        self._relations = []
        self._relation_to_ind = {}  # start from 1
        with open(osp.join(self.vocab_dir, 'relations_vocab.txt')) as f:
            count = 1
            for rel in f.readlines():
                names = [n.lower().strip() for n in rel.split(',')]
                self._relations.append(names[0])
                for n in names:
                    self._relation_to_ind[n] = count
                count += 1      

        # image_ids for this split
        tic = time.time()
        self._image_index = self._load_image_set_index()
        print(f'{len(self._image_index)} image_ids for {image_set} prepared '
              f'in {time.time()-tic:.2f} seconds.')

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_index = []
        for img in self.image_data[self._image_set]:
            image_id = img['id']
            xml_path = self._annotation_xml_path(image_id)
            if os.path.exists(xml_path):
                # Some images have no bboxes after object filtering, so there
                # is no xml annotation for these.
                tree = ET.parse(xml_path)
                for obj in tree.findall('object'):
                    obj_name = obj.find('name').text.lower().strip()
                    if obj_name in self._class_to_ind:
                        # We have to actually load and check these to make sure 
                        # they have at least one object actually in vocab
                        image_index.append(image_id)
                        break
        print(f'{len(image_index)} out of '
              f'{len(self.image_data[self._image_set])} are selected.')
        return image_index

    def _annotation_xml_path(self, index):
        return os.path.join(self.genome_dir, 'xml', str(index) + '.xml')

    def load_vg_annotations(self):
        """
        Load image and bounding boxes info from XML file in the COCO instances
        format.
        - info   : {description, url, version, year}
        - images : [{file_name, coco_id, flickr_id, height, width, id}]
        - annotations: [{area, iscrowd, image_id, bbox, category_id, id}]
        - categories : [{supercategory, id, name}]
        """
        annotations = []
        ann_id = 0
        for index in self._image_index:
            # image data
            image_meta = self.iid_to_meta[index] 
            height, width = image_meta['height'], image_meta['width']

            # xml annotations for this image
            xml_path = self._annotation_xml_path(index)
            tree = ET.parse(xml_path)
            objs = tree.findall('object')

            # add objects within this image
            for obj in objs:
                obj_name = obj.find('name').text.lower().strip()
                if obj_name in self._class_to_ind:
                    bbox = obj.find('bndbox')
                    x1 = max(0,float(bbox.find('xmin').text))
                    y1 = max(0,float(bbox.find('ymin').text))
                    x2 = min(width-1,float(bbox.find('xmax').text))
                    y2 = min(height-1,float(bbox.find('ymax').text))
                    # If bboxes are not positive, just given whole image coords
                    # there are a few such samples
                    if x2 < x1 or y2 < y1:
                        print(f'Failed bbox in {xml_path}, object {obj_name}')
                        x1 = 0
                        y1 = 0
                        x2 = width - 1
                        y2 = height - 1
                    bbox = [x1, y1, x2-x1+1, y2-y1+1]
                    # attribute category
                    att_class_ids = []
                    n = 0
                    for att in obj.findall('attribute'):
                        att = att.text.lower().strip()
                        if att in self._attribute_to_ind:
                            att_class_ids.append(self._attribute_to_ind[att])
                            n += 1
                        if n >= 16:
                            break
                    # add to annotations
                    ann = {
                        'id': ann_id,
                        'category_id': self._class_to_ind[obj_name],
                        'image_id': index,
                        'bbox': bbox,
                        'attribute_ids': att_class_ids,
                    }
                    annotations.append(ann)
                    # next anno
                    ann_id += 1
        # return
        print(f'{len(annotations)} objects found for '
              f'{len(self._image_index)} [{self._image_set}] images.')
        return annotations

    def load_vg_categories(self):
        categories = []
        for obj_cls, cls_id in self._class_to_ind.items():
            cat = {'name': obj_cls, 'id': cls_id}
            categories.append(cat)
        return categories

    def load_vg_attributes(self):
        attributes = []
        for att_cls, cls_id in self._attribute_to_ind.items():
            att = {'name': att_cls, 'id': cls_id}
            attributes.append(att)
        return attributes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--genome_dir', type=str, 
                        default='datasets/genome')
    parser.add_argument('--vocab_dir', type=str, 
                        default='datasets/genome/1600-400-20')
    parser.add_argument('--output_dir', type=str, 
                        default='datasets/genome/annotations')
    parser.add_argument('--filter', dest='filter', 
                        action='store_true')
    args = parser.parse_args()

    if not osp.isdir(args.output_dir): os.makedirs(args.output_dir)

    # make instances
    for split in ['train', 'val', 'test']:
        
        dataset = vg(args.genome_dir, args.vocab_dir, split, args.filter)
        images = dataset.image_data[split]
        annotations = dataset.load_vg_annotations()
        categories = dataset.load_vg_categories()
        attributes = dataset.load_vg_attributes()

        # write
        output_file = f'instances_{split}.json' if not args.filter \
                    else f'filtered_instances_{split}.json'
        output_file = osp.join(args.output_dir, output_file)
        with open(output_file, 'w') as f:
            json.dump({
                'images': images, 
                'annotations': annotations,
                'attributes': attributes, 
                'categories': categories
            }, f)
        print(f'{output_file} written.')
