export TOKENIZERS_PARALLELISM=false
python -u /your_path/SpikeCLIP/vision/no_LSD/pretrain.py \
    --device "cuda:x" \
    --seed 42 \
    --clip_name "openai/clip-vit-base-patch16" \
    --clip_dim 512 \
    --embed_dims 384 \
    --in_channels 3 \
    --num_heads 8 \
    --mlp_ratios 4 \
    --depths 4 \
    --T 4 \
    --language_path "/text_encoder_path/model_best.pth.tar" \
    --batchsize 1 \
    --num_workers 4 \
    --dataset 'OxfordIIITPet' \
    --lr 5e-3 \
    --weight_decay 0\
    --amsgrad True \
    --T_max 200 \
    --eta_min 1e-4 \
    --eval_metric 'top1' \
    --max_history 3 \
    --epoches 200 \
    > /your_path/SpikeCLIP/vision/no_LSD/pre_OxfordIIITPet.log