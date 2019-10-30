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

