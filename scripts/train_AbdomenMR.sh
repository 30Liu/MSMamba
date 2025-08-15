

MAMBA_MODEL=$1
PRED_OUTPUT_PATH="/dataset/liuruoyun/data/nnUNet_results/Dataset702_AbdomenMR/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="/dataset/liuruoyun/data/nnUNet_results/Dataset702_AbdomenMR/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="5,6,7"

# train
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 702 2d all -tr ${MAMBA_MODEL} -num_gpus 3  &&

echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "/dataset/liuruoyun/data/nnUNet_raw/Dataset702_AbdomenMR/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 702 \
    -c 2d \
    -npp 1 \
    -nps 1 \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_best.pth" &&

echo "Computing dice..."
python evaluation/abdomen_DSC_Eval.py \
    --gt_path "/dataset/liuruoyun/data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_DSC.csv"  &&

echo "Computing NSD..."
python evaluation/abdomen_NSD_Eval.py \
    --gt_path "/dataset/liuruoyun/data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_NSD.csv" &&

echo "Done."

#CUDA_VISIBLE_DEVICES="1" nnUNetv2_predict \
#    -i "/dataset/liuruoyun/data/nnUNet_raw/Dataset702_AbdomenMR/imagesTs" \
#    -o "/dataset/liuruoyun/data/nnUNet_results/Dataset702_AbdomenMR/nnUNetTrainerUNETR__nnUNetPlans__2d/pred_results" \
#    -d 702 \
#    -c 2d \
#    -npp 1 \
#    -nps 1 \
#    -tr "nnUNetTrainerUNETR" \
#    --disable_tta \
#    -f all \
#    -chk "checkpoint_best.pth"
####
#python evaluation/abdomen_DSC_Eval.py \
#    --gt_path "/dataset/liuruoyun/data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs" \
#    --seg_path "/dataset/liuruoyun/data/nnUNet_results/Dataset702_AbdomenMR/nnUNetTrainerSwinUMamba__nnUNetPlans__2d/pred_results" \
#    --save_path "/dataset/liuruoyun/data/nnUNet_results/Dataset702_AbdomenMR/nnUNetTrainerSwinUMamba__nnUNetPlans__2d/metric_DSC.csv"
#
#
#python evaluation/abdomen_NSD_Eval.py \
#    --gt_path "/dataset/liuruoyun/data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs" \
#    --seg_path "/dataset/liuruoyun/data/nnUNet_results/Dataset702_AbdomenMR/nnUNetTrainerSwinUMamba__nnUNetPlans__2d/pred_results" \
#    --save_path "/dataset/liuruoyun/data/nnUNet_results/Dataset702_AbdomenMR/nnUNetTrainerSwinUMamba__nnUNetPlans__2d/metric_NSD.csv"