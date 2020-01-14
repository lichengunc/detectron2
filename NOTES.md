## Evaluate Detectron2 Models
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

## Train and Evaluate Genome Object+Attribute Detector
- Run ```./experiments/genome_faster_rcnn_softmax_attr_R_101_C4_3x.sh```.
It trains ResNet-101 C4 for 1600 object classes and 400 attribute classes, with evaluation supported.
- Run ```./experiments/evaluate_c4_feats.sh``` to extract all COCO images' features
and use [mcan-vqa](https://github.com/MILVLG/mcan-vqa) to test the performance.

## Some Ipython Notebooks to Check Results
- ```check_vg_data.ipynb```: check statistics of VG data.
- ```run_attr_inference.ipynb```: run attr+obj detection together and visualize the results.
- ```make_json_file_for_extract_feats_given_boxes.ipynb```: make temporary json file for testing _extract_feats_given_boxes_.
- ```check_extract_feats_given_boxes.ipynb```: visualize results of _extract_feats_given_boxes_ after running ```python tools/extract_feats_given_boxes.py```.
- ```check_extract_feats.ipynb```: visualize results of _extract_feats_ after running ```python tools/extract_feats.py```.

## Results
Surprisingly, C4 works better than FPN in terms of both detection accuracy and downstream VQA evaluation.
Here, I list the C4 results for reference.

<table>
<th>Detection Results</th>
<tr><td>

| | mAP | AP50 | AP50 (attr) | weighted AP50 (attr) |
|--|--|--|--|--|
| [butd](https://github.com/peteanderson80/bottom-up-attention) | - | 10.2 | 7.8 | 27.8 |
| Res101-FPN | 5.3215 | 10.4593 | 8.23 | 26.22 |
| Res101-C4  | 5.5161 | 10.7825 | 8.99 | 26.80 |
</td></tr> </table>

<table>
<th> mcan-vqa Results</th>
<tr><td>

| Features used by mcan | VQA Accuracy (val) |
|--|--|
| [butd](https://github.com/peteanderson80/bottom-up-attention) | 67.15 |
| Res101-FPN (nms0.30_conf0.20_max100_min10) | 66.56 |
| Res101-FPN (nms0.30_conf0.15_max100_min10) | 66.84 |
| Res101-FPN (nms0.30_conf0.10_max100_min10) | 67.08 |
| Res101-C4 (nms0.30_conf0.20_max100_min10)  | 67.52 |
| Res101-C4 (nms0.30_conf0.15_max100_min10)  | 67.73 |
| Res101-C4 (nms0.30_conf0.10_max100_min10)  | 67.86 |
| Res101-C4 (nms0.50_conf0.20_max100_min10)  | 67.67 |
| Res101-C4 (nms0.50_conf0.10_max100_min10)  | 67.xx |
</td></tr> </table>