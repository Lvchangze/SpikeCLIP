export TOKENIZERS_PARALLELISM=false
python -u -m torch.distributed.launch --nproc_per_node=x --nnodes=x --node_rank=0 --master_addr="xxxx" --master_port=xxxx xxx/alignment.py \
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
    --batchsize 196 \
    --num_workers 4 \
    --dataset 'ImageNet_1k' \
    --lr 5e-3 \
    --weight_decay 0\
    --amsgrad True \
    --T_max 50 \
    --eta_min 5e-4 \
    --eval_metric 'top1' \
    --max_history 1 \
    --epoches 50 \
    > /your_path/SpikeCLIP/vision/alignment_ft/alignment_50epoch.sh
