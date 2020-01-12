# feature extraction
config_file="configs/GENOME-Detection/faster_rcnn_softmax_attr_R_101_C4_3x.yaml"
model_weights="output/genome_faster_rcnn_softmax0.5_expminus_attr_R_101_C4_3x/model_final.pth"
nms_thresh=0.30
score_thresh=0.10
max_dets=100
min_dets=10
enforce_topk=false
gpu=2

# run
for split in train2014 val2014
do
        if [ "$enforce_topk" = false ]; then
                python tools/extract_feats.py \
                        --config-file ${config_file} \
                        --model-weights ${model_weights} \
                        --image-root "datasets/coco/${split}" \
                        --nms-thresh ${nms_thresh} \
                        --score-thresh ${score_thresh} \
                        --max-detections ${max_dets} \
                        --min-detections ${min_dets} \
                        --output-dir "output/coco_c4_feats/nms${nms_thresh}_conf${score_thresh}_max${max_dets}_min${min_dets}/${split}"
        else
                python tools/extract_feats.py \
                        --config-file ${config_file} \
                        --model-weights ${model_weights} \
                        --image-root "datasets/coco/${split}" \
                        --nms-thresh ${nms_thresh} \
                        --score-thresh ${score_thresh} \
                        --max-detections ${max_dets} \
                        --min-detections ${min_dets} \
                        --output-dir "output/coco_c4_feats/nms${nms_thresh}_conf${score_thresh}_enforce${max_dets}/${split}" \
                        --enforce-topk
        fi
done

# run vqa
cd projects/mcan-vqa && \
python run.py --RUN='train' \
              --GPU ${gpu} \
              --IMG_FEAT_SIZE=2048 \
              --VERSION="c4_nms${nms_thresh}_conf${score_thresh}_max${max_dets}_min${min_dets}" \
              --FEAT_PATH="../../output/coco_c4_feats/nms${nms_thresh}_conf${score_thresh}_max${max_dets}_min${min_dets}/"

