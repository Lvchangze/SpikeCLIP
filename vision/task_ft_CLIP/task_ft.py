import argparse
import time
from datetime import datetime
import logging
import os
import yaml
from collections import OrderedDict
from contextlib import suppress
import copy
from tqdm import tqdm
import importlib
import random
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import grad
import math

from transformers import CLIPProcessor, CLIPModel
from spikingjelly.clock_driven import functional
from timm.utils import get_outdir, CheckpointSaver

from ..utils import *
from ..model import language_encoder, vision_encoder
from ..clip_eval import my_load_dataset, load_txt_dataset





def args():
    parser = argparse.ArgumentParser(description='CLIP with Task Specificly FT', add_help=False)
    
    parser.add_argument('--device', default="cuda:0", type=str, help=None)
    parser.add_argument('--seed', default=42, type=int, help=None)
    parser.add_argument('--clip_name', default="openai/clip-vit-base-patch16", type=str, help=None)
    parser.add_argument('--clip_dim', default=512, type=int, help=None)
    parser.add_argument('--mlp_ratios', default=4, type=int, help=None)
    parser.add_argument('--depths', default=4, type=int, help=None)
    parser.add_argument("--T", default=1, type=int, help=None)
    parser.add_argument('--language_path', default=None, type=str, help=None)

    parser.add_argument("--batchsize", default=32, type=int, help=None)
    parser.add_argument("--num_workers", default=4, type=int, help=None)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help=None)

    parser.add_argument("--lr", default=5e-5, type=float, help=None)
    parser.add_argument("--betas", default=(0.9, 0.999), type=tuple, help=None)
    parser.add_argument("--weight_decay", default=0., type=float, help=None)
    parser.add_argument("--amsgrad", default=False, type=bool, help=None)

    parser.add_argument('--T_max', default=200, type=int, help=None)
    parser.add_argument('--eta_min', default=2e-8, type=float, help=None)
    parser.add_argument('--epoches', default=2000, type=int, help=None)

    args = parser.parse_args()
    return args


def main(args):
    clip = CLIPModel.from_pretrained(args.clip_name).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_name)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    clip_embedding = copy.deepcopy(clip.text_model.embeddings)

    module_name = f"label_sentence.{args.dataset}"
    module = importlib.import_module(module_name)
    classes = module.classes
    templates = module.templates
    train_dataset, test_dataset = my_load_dataset(args.dataset, image_processor, image_processor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False)
    
    txt_model = language_encoder(
        pretrained=False,
        clip_embedding=clip_embedding,
        clip_dim=512,
        mlp_ratios=4,
        depths=4,
        T=4,
    ).to(args.device)

    checkpoint = torch.load(args.language_path)
    txt_model.load_state_dict(checkpoint['state_dict'])
    labels_candidate = labels_candidates(classes, templates, txt_model, tokenizer).to(args.device)
    labels_candidate.requires_grad = False
    optimizer = optim.Adam(clip.parameters(), lr=args.lr, betas=args.betas, eps=1e-08, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    
    best_top1=0.
    best_epoch=0
    for epoch in tqdm(range(args.epoches)):
        train_losses = 0.
        clip.train()
        for batch_idx, (images, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                images = images['pixel_values'][0].to(args.device)
                target = target.to(args.device)
                clip_features = clip.get_image_features(images)
                clip_features = F.normalize(clip_features, p=2, dim=-1)
                logits = clip_features @ labels_candidate
                loss = nn.CrossEntropyLoss()(logits, target)
                train_losses += loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        clip.eval()
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images = images['pixel_values'][0].to(args.device)
                target = target.to(args.device)
                clip_features = clip.get_image_features(images)
                clip_features = F.normalize(clip_features, p=2, dim=-1)
                logits = clip_features @ labels_candidate

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

            # optput ACC
            top1 = (top1 / n) * 100
            top5 = (top5 / n) * 100 
            if top1>best_top1:
                best_top1=top1
                best_epoch=epoch
                torch.save(clip, f'/data/ltlong/snnclip/output/{args.dataset}/clip_{top1}_{epoch}.pth')
        print(f"\n>>>>>  Epoch:[{epoch}/{args.epoches}]----> train_losses:{train_losses}, current_lr:{current_lr:.4e}, top1_acc:{top1:.4f}%, top5_acc:{top5:.4f}%, best_top1:{best_top1}, best_epoch:{best_epoch}\n")


if __name__ == '__main__':
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    
    set_seed(_args.seed)
    main(_args)









