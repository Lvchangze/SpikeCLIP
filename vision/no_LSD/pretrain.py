import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import datetime
import argparse
import importlib
import random
import numpy as np
import copy
from tqdm import tqdm
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
from timm.utils import get_outdir, CheckpointSaver
from spikingjelly.clock_driven import functional

from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from ..utils import *
from ..model import language_encoder, vision_encoder
from clip_eval import my_load_dataset, load_txt_dataset



def args():
    parser = argparse.ArgumentParser(description='Training Vision-Encoder without LSD', add_help=False)
    
    parser.add_argument('--device', default="cuda:0", type=str, help=None)
    parser.add_argument('--seed', default=42, type=int, help=None)
    parser.add_argument('--clip_name', default="openai/clip-vit-base-patch16", type=str, help=None)
    parser.add_argument('--clip_dim', default=512, type=int, help=None)
    parser.add_argument('--embed_dims', default=384, type=int, help=None)
    parser.add_argument('--in_channels', default=3, type=int, help=None)
    parser.add_argument('--num_heads', default=8, type=int, help=None)
    parser.add_argument('--mlp_ratios', default=4, type=int, help=None)
    parser.add_argument('--depths', default=4, type=int, help=None)
    parser.add_argument("--T", default=4, type=int, help=None)
    parser.add_argument('--language_path', default=None, type=str, help=None)

    parser.add_argument("--batchsize", default=4, type=int, help=None)
    parser.add_argument("--num_workers", default=4, type=int, help=None)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help=None)

    parser.add_argument("--lr", default=5e-5, type=float, help=None)
    parser.add_argument("--betas", default=(0.9, 0.999), type=tuple, help=None)
    parser.add_argument("--weight_decay", default=0., type=float, help=None)
    parser.add_argument("--amsgrad", default=False, type=bool, help=None)

    parser.add_argument('--T_max', default=200, type=int, help=None)
    parser.add_argument('--eta_min', default=2e-8, type=float, help=None)

    parser.add_argument('--eval_metric', default='top1', type=str, help=None)
    parser.add_argument('--max_history', default=5, type=int, help=None)
    parser.add_argument('--epoches', default=2000, type=int, help=None)

    args = parser.parse_args()
    return args



def train(
    epoches=2,
    model=None,
    clip_model=None,
    train_loader=None,
    test_loader=None, 

    batchsize=4,
    optimizer=None,
    scaler=None,
    lr_scheduler=None,

    labels_candidate=None,
    saver=None,
    device=None
):  
    for epoch in tqdm(range(epoches)):
        # #############################################################################################################
        # ###################################             Train            ############################################
        # #############################################################################################################
        model.train()
        train_losses = 0.
        for batch_idx, (images, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                images = images['pixel_values'][0].to(device)
                resized_img = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
                model_features = model(resized_img)
                clip_features = clip_model.module.get_image_features(images)
                loss = cosine_similarity_loss(model_features, clip_features)
                train_losses += loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(model)
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
        # #############################################################################################################
        # ###################################           Validation         ############################################
        # #############################################################################################################
        model.eval()
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images = images['pixel_values'][0].to(device)
                target = target.to(device)
                resized_img = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
                image_features = model(resized_img)
                image_features = F.normalize(image_features, p=2, dim=-1)
                logits = image_features @ labels_candidate
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
                functional.reset_net(model)

            # optput ACC
            top1 = (top1 / n) * 100
            top5 = (top5 / n) * 100 
            best_top1, best_epoch = saver.save_checkpoint(epoch, metric=top1)
        print(f"\n>>>>>  Epoch:[{epoch+1}/{epoches}]----> train_loss:{train_losses}, lr:{current_lr:.2e}, top1_acc:{top1:.4f}%, top5_acc:{top5:.4f}%, best_top1:{best_top1:.4f}%, best_epoch:{best_epoch}")
        print(f"Current time: {datetime.datetime.now()}\n")


def main(args):
    # classes & templates
    module_name = f"label_sentence.{args.dataset}"
    module = importlib.import_module(module_name)
    classes = module.classes
    templates = module.templates
    print(f">>>>>>>>>>   The evaluation  will be performe on {args.dataset}   <<<<<<<<<<")
    print(f"{len(classes)} classes, {len(templates)} templates")

    # CLIP
    clip_model = CLIPModel.from_pretrained(args.clip_name).to(args.device)
    for param in clip_model.parameters():
        param.requires_grad = False
    processor = CLIPProcessor.from_pretrained(args.clip_name)
    tokenizer = processor.tokenizer
    img_processor = processor.image_processor
    clip_embedding = copy.deepcopy(clip_model.text_model.embeddings)
    
    # vision
    model = vision_encoder(
        in_channels=args.in_channels,
        target_dim=512,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        mlp_ratios=args.mlp_ratios,
        depths=args.depths,
        T=args.T
    ).to(args.device)
    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    # text
    txt_model = language_encoder(
        pretrained=False,
        clip_embedding=clip_embedding,
        clip_dim=args.clip_dim,
        mlp_ratios=args.mlp_ratios,
        depths=args.depths,
        T=args.T
    ).to(args.device)
    checkpoint = torch.load(args.language_path)
    txt_model.load_state_dict(checkpoint['state_dict'])
    labels_candidate = labels_candidates(classes, templates, txt_model, tokenizer).to(args.device)
    labels_candidate.requires_grad = False

    # parallel
    if torch.cuda.is_available():
        device_ids = [i for i in range(torch.cuda.device_count())]
        if len(device_ids) > 1:
            clip_model = nn.DataParallel(clip_model, device_ids=device_ids)
            model = nn.DataParallel(model, device_ids=device_ids)
    

    # dataset
    trainset, testset =my_load_dataset(dataset_name=args.dataset, transform_train=img_processor, transform_test=img_processor)
    concat_dataset = my_load_dataset(dataset_name="ImageNet_1k", transform_train=img_processor, transform_test=img_processor)[0]

    train_loader = DataLoader(
        concat_dataset,
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        shuffle=True
    )
 
    test_loader = DataLoader(
        testset,
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        shuffle=False
    ) 

    # prepare for Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=1e-08, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"),])
    output_dir = get_outdir(f'./output/{args.dataset}', exp_name)
    eval_metric = args.eval_metric
    decreasing = True if eval_metric == 'loss' else False

    saver = CheckpointSaver(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=output_dir,
        recovery_dir=output_dir,
        decreasing=decreasing,
        max_history=args.max_history
    )

    train(
        epoches=args.epoches,
        model=model,
        clip_model=clip_model,
        train_loader=train_loader,
        test_loader=test_loader,
        batchsize=args.batchsize,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=scheduler,
        labels_candidate=labels_candidate,
        saver=saver,
        device=args.device
    )


if __name__ == '__main__':
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    
    set_seed(_args.seed)
    main(_args)
