## Evaluation
We could change the config file by replacing `detectron2://ImageNetPretrained/MSRA/R-50.pkl` with `output/model_final.pth`.

Or we could do evaluation using command line:
```
python tools/train_net.py \
        --num-gpus 8 \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
        --eval-only \
        'MODEL.WEIGHTS' 'output/model_final.pth' 
```
where `eval-only` denotes evaluation mode and `MODEL.WEIGHTS` are the trained model path.

## Prepare COCO-style annotations for Visual Genome
Download image splits and 1600-400-20 vocab for Genome, provided by Peter Anderson.
```
./datasets/prepare_for_genome.sh
```
Find image_ids to be excluded (filtered out) by running
```
python tools/excluded_coco_vg_iids.py
```
Then create vg_splits, ```filter``` controls if we want to exclude the image_ids in Refer's val/test split and Flickr30K.
```
python create_vg_splits.py --filter
```
Then make xml annotations for each image
```
python tools/setup_vg.py
```
Finally make COCO-style annotations, ```filter``` controls if we want to filter out images in REFER and Flickr30K.
``` 
python tools/make_vg_instances.py --filter
```
Also make meta.json for genome dataset, as we want to load the meta information during inference time.
For meta information, we only run statistics on ```train``` portion.
```
python tools/make_vg_meta.py
```
