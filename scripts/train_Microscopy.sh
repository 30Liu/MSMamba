# V1 can load pretrained model of VSS Block

MAMBA_MODEL=$1
PRED_OUTPUT_PATH="/dataset/liuruoyun/data/nnUNet_results/Dataset703_NeurIPSCell/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="/dataset/liuruoyun/data/nnUNet_results/Dataset703_NeurIPSCell/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="1"

# train
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 703 2d all -tr ${MAMBA_MODEL}  &&

# predictq
echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "/dataset/liuruoyun/data/nnUNet_raw/Dataset703_NeurIPSCell/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 703 \
    -c 2d \
    -npp 1 \
    -nps 1 \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_best.pth" &&

echo "Computing F1..."
python evaluation/compute_cell_metric.py \
    --gt_path "/dataset/liuruoyun/data/nnUNet_raw/Dataset703_NeurIPSCell/labelsTs" \
    -s "${PRED_OUTPUT_PATH}" \
    -o "${EVAL_METRIC_PATH}" \
    -n "${MAMBA_MODEL}_703_2d"  &&

echo "Done."

#CUDA_VISIBLE_DEVICES='7' nnUNetv2_predict \
#    -i "/dataset/liuruoyun/data/nnUNet_raw/Dataset703_NeurIPSCell/imagesTs" \
#    -o "/dataset/liuruoyun/data/nnUNet_results/Dataset703_NeurIPSCell/nnUNetTrainerUNETR__nnUNetPlans__2d/pred_results"\
#    -d 703 \
#    -c 2d \
#    -npp 1 \
#    -nps 1 \
#    -tr "nnUNetTrainerUNETR" \
#    --disable_tta \
#    -f all \
#    -chk "checkpoint_best.pth"

#python evaluation/compute_cell_metric.py \
#    --gt_path "/dataset/liuruoyun/data/nnUNet_raw/Dataset703_NeurIPSCell/labelsTs" \
#      -s "/dataset/liuruoyun/data/nnUNet_results/Dataset703_NeurIPSCell/nnUNetTrainerSwinUMamba__nnUNetPlans__2d/pred_results" \
#    -o "/dataset/liuruoyun/data/nnUNet_results/Dataset703_NeurIPSCell/nnUNetTrainerSwinUMamba__nnUNetPlans__2d" \
#    -n "nnUNetTrainerSwinUMamba_703_2d"