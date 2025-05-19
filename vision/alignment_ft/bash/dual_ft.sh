export TOKENIZERS_PARALLELISM=false
python -u -m torch.distributed.launch --nproc_per_node=x --nnodes=x --node_rank=0 --master_addr="xxxx" --master_port=xxxx xxx/dual_ft.py \
    --device "cuda:x" \
    --seed 42 \
    --clip_name "openai/clip-vit-base-patch16" \
    --clip_path "Task_FT_CLIP_dataset_name_path" \
    --clip_dim 512 \
    --embed_dims 384 \
    --in_channels 3 \
    --num_heads 8 \
    --mlp_ratios 4 \
    --depths 4 \
    --T 4 \
    --vision_path "/alignment_model/model_best.pth.tar" \
    --language_path "/text_encoder_path/model_best.pth.tar" \
    --batchsize 196 \
    --num_workers 4 \
    --datase 'dataset_name' \
    --lr 5e-4 \
    --weight_decay 0\
    --amsgrad True \
    --T_max 200 \
    --eta_min 1e-5 \
    --eval_metric 'top1' \
    --max_history 1 \
    --epoches 200 \
    --boundary_angle -90 \
    > /your_path/SpikeCLIP/vision/alignment_ft/ft_datsetset_name.log
