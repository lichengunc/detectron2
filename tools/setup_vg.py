''' Visual genome data analysis and preprocessing.'''
import json
import os
import os.path as osp
import operator
from collections import Counter, defaultdict
import argparse

# Set common attributes
common_attributes = set(['white','black','blue','green','red','brown','yellow',
    'small','large','silver','wooden','orange','gray','grey','metal','pink',
    'tall','long','dark'])

"""
Add attributes to `scene_graph.json`, extracted from `attributes.json`.
This also adds a unique id to each attribute, and separates individual
attibutes for each object (these are grouped in `attributes.json`).
"""
def SceneGraphsWithAttrs(dataDir='datasets/vg'):
    attr_data = json.load(open(os.path.join(dataDir, 'attributes.json')))
    with open(os.path.join(dataDir, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}

    id_count = 0
    for img_attrs in attr_data:
        attrs = []
        for attribute in img_attrs['attributes']:
            a = img_attrs.copy(); del a['attributes']
            a['attribute']    = attribute
            a['attribute_id'] = id_count
            attrs.append(a)
            id_count += 1
        iid = img_attrs['image_id']
        sg_dict[iid]['attributes'] = attrs
    
    return sg_dict.values()

def clean_string(string):
    string = string.lower().strip()
    if len(string) >= 1 and string[-1] == '.':
        return string[:-1].strip()
    return string

def clean_objects(string, common_attributes):
    ''' Return object and attribute lists '''
    string = clean_string(string)
    words = string.split()
    if len(words) > 1:
        prefix_words_are_adj = True
        for att in words[:-1]:
            if not att in common_attributes:
                prefix_words_are_adj = False
        if prefix_words_are_adj:
            return words[-1:], words[:-1]
        else:
            return [string], []
    else:
        return [string], []
    
def clean_attributes(string):
    ''' Return attribute list '''
    string = clean_string(string)
    if string == "black and white":
        return [string]
    else:
        return [word.lower().strip() for word in string.split(" and ")]

def clean_relations(string):
    string = clean_string(string)
    if len(string) > 0:
        return [string]
    else:
        return []

def build_vocabs(args):
    objects = Counter()
    attributes = Counter()
    relations = Counter()

    data = SceneGraphsWithAttrs(args.data_dir)

    # First extract attributes and relations
    for sg in data:
        for attr in sg['attributes']:
            try:
                attributes.update(
                        clean_attributes(attr['attribute']['attributes'][0])
                    )
            except:
                pass
        for rel in sg['relationships']:
            relations.update(clean_relations(rel['predicate']))

    # Now extract objects, while looking for common adjectives that will be 
    # repurposed as attributes
    for sg in data:
        for obj in sg['objects']:
            o, a = clean_objects(obj['names'][0], common_attributes)
            objects.update(o)
            attributes.update(a)

    # Create full-sized vocabs
    objects = set([k for k,v in objects.most_common(args.max_objects)])
    attributes = set([k for k,v in attributes.most_common(args.max_attributes)])
    relations = set([k for k,v in relations.most_common(args.max_relations)])

    with open(
        osp.join(args.output_dir, f'objects_vocab_{args.max_objects}.txt'), 
        'w') as txt_file:
        for item in objects:
            txt_file.write(f'{item}\n')

    with open(
        osp.join(args.output_dir, f'attributes_vocab_{args.max_attributes}.txt'), 
        'w') as txt_file:
        for item in attributes:
            txt_file.write(f'{item}\n')

    with open(
        osp.join(args.output_dir, f'relations_vocab_{args.max_relations}.txt'), 
        'w') as txt_file:
        for item in relations:
            txt_file.write(f'{item}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/vg')
    parser.add_argument('--max_objects', type=int, default=1600)
    parser.add_argument('--max_attributes', type=int, default=400)
    parser.add_argument('--max_relations', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='datasets/prepro')
    args = parser.parse_args()
    build_vocabs(args)
