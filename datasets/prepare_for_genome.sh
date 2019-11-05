#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Download some files needed for running tests.

cd "${0%/*}"

mkdir -p genome
mkdir -p genome/image_splits
mkdir -p genome/1600-400-20

# download image_ids
wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/train.txt -P genome/image_splits
wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/val.txt -P genome/image_splits
wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/test.txt -P genome/image_splits

# download vocab for 1600 objects + 400 attributes + 20 relations 
wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/1600-400-20/objects_vocab.txt -P genome/1600-400-20
wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/1600-400-20/attributes_vocab.txt -P genome/1600-400-20
wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/1600-400-20/relations_vocab.txt -P genome/1600-400-20


