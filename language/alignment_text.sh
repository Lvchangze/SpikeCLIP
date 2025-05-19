export TOKENIZERS_PARALLELISM=false
sudo /your_path/python -u /your_path/SpikeCLIP/language/text_encoder.py \
    --device "x" \
    --seed 42 \
    --clip_name "openai/clip-vit-base-patch16" \
    --embed_dims 512 \
    --num_heads 8 \
    --mlp_ratios 4 \
    --depths 4 \
    --T 4 \
    --checkpoint_path "" \
    --lr 5e-3 \
    --weight_decay 0 \
    --amsgrad True \
    --T_max 2000 \
    --eta_min 5e-4 \
    --eval_metric 'loss' \
    --max_history 3 \
    --eval_dataset_name 'CIFAR10' \
    --epoches 2000 \
    --batch_size 1024 \
    > /your_path/SpikeCLIP/language/log/text_encoder.log



