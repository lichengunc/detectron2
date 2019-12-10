# training
python tools/train_net.py \
        --num-gpus 8 \
        --config-file configs/GENOME-Detection/faster_rcnn_attr_R_50_FPN_1x.yaml 

# # evaluation
# python tools/train_net.py \
#         --num-gpus 8 \
#         --config-file configs/GENOME-Detection/faster_rcnn_attr_R_50_FPN_1x.yaml \
#         --eval-only \
#         'MODEL.WEIGHTS' 'output/genome_faster_rcnn_R_50_RPN_1x/model_final.pth' 