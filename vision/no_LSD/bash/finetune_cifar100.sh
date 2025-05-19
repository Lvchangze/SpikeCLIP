export TOKENIZERS_PARALLELISM=false
python -u /your_path/SpikeCLIP/vision/no_LSD/finetune.py \
    --device "cuda:x" \
    --seed 42 \
    --clip_name "openai/clip-vit-base-patch16" \
    --clip_path "Task_FT_CLIP_cifar100_path" \
    --vision_path "pretrained/model_best.pth.tar" \
    --language_path "/text_encoder_path/model_best.pth.tar" \
    --in_channels 3 \
    --clip_dim 512 \
    --embed_dims 384 \
    --num_heads 8 \
    --mlp_ratios 4 \
    --depths 4 \
    --T 4 \
    --batchsize 32 \
    --num_workers 4 \
    --dataset 'CIFAR100' \
    --boundary_angle 90 \
    --lr 5e-4 \
    --weight_decay 0\
    --amsgrad True \
    --T_max 200 \
    --eta_min 5e-5 \
    --epoches 200 \
    > /your_path/SpikeCLIP/vision/no_LSD/ft_cifar100.log