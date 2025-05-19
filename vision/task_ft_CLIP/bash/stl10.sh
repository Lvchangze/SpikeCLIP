export TOKENIZERS_PARALLELISM=false
python -u /your_path/SpikeCLIP/vision/task_ft_CLIP/task_ft.py \
    --device "cuda:x" \
    --seed 42 \
    --clip_name "openai/clip-vit-base-patch16" \
    --clip_dim 512 \
    --mlp_ratios 4 \
    --depths 1 \
    --T 4 \
    --language_path "/text_encoder_path/model_best.pth.tar" \
    --batchsize 32 \
    --num_workers 4 \
    --dataset 'STL10' \
    --lr 1e-6 \
    --weight_decay 0\
    --amsgrad True \
    --T_max 20 \
    --eta_min 1e-8 \
    --epoches 20 \
    > /your_path/SpikeCLIP/vision/task_ft_CLIP/log/stl10.log