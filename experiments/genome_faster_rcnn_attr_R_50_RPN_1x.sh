attr_loss_weight=0.75
attr_loss_type="softmax"
attr_sampling="expminus"

# training
python tools/train_net.py \
        --num-gpus 8 \
        --config-file configs/GENOME-Detection/faster_rcnn_${attr_loss_type}_attr_R_50_FPN_1x.yaml \
        "MODEL.ROI_HEADS.ATTRIBUTE_LOSS_TYPE" ${attr_loss_type} \
        "MODEL.ROI_HEADS.ATTRIBUTE_LOSS_WEIGHT" ${attr_loss_weight} \
        "MODEL.ROI_HEADS.ATTRIBUTE_SAMPLING" ${attr_sampling} \
        "OUTPUT_DIR" "output/genome_faster_rcnn_${attr_loss_type}${attr_loss_weight}_${attr_sampling}_attr_R_50_FPN_1x"

# evaluation
python tools/train_net.py \
        --num-gpus 8 \
        --config-file configs/GENOME-Detection/faster_rcnn_${attr_loss_type}_attr_R_50_FPN_1x.yaml \
        --eval-only \
        "MODEL.WEIGHTS" "output/genome_faster_rcnn_${attr_loss_type}${attr_loss_weight}_${attr_sampling}_attr_R_50_FPN_1x/model_final.pth"

# attribute evaluation
python tools/train_net.py \
        --num-gpus 8 \
        --config-file configs/GENOME-Detection/faster_rcnn_${attr_loss_type}_attr_R_50_FPN_1x.yaml \
        --eval-only \
        --eval-attribute \
        "MODEL.WEIGHTS" "output/genome_faster_rcnn_${attr_loss_type}${attr_loss_weight}_${attr_sampling}_attr_R_50_FPN_1x/model_final.pth"

