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
from clip_eval import my_load_dataset, load_txt_dataset






def args():
    parser = argparse.ArgumentParser(description='Fine-tune Vision-Encoder without LSD', add_help=False)
    
    parser.add_argument('--device', default="cuda:0", type=str, help=None)
    parser.add_argument('--seed', default=42, type=int, help=None)
    parser.add_argument('--clip_name', default="openai/clip-vit-base-patch16", type=str, help=None)
    parser.add_argument('--clip_path', default="/data/ltlong/snnclip/output/Caltech101/clip_94.47785675232367_21.pth", type=str, help=None)
    parser.add_argument('--vision_path', default="/data/ltlong/snnclip/output/CIFAR10/completed_386_88.61/model_best.pth.tar", type=str, help=None)
    parser.add_argument('--language_path', default="/data/ltlong/snnclip/output/txt_encoder/20230623-082503/model_best.pth.tar", type=str, help=None)
    
    parser.add_argument('--in_channels', default=3, type=int, help=None)
    parser.add_argument('--clip_dim', default=512, type=int, help=None)
    parser.add_argument('--embed_dims', default=384, type=int, help=None)
    parser.add_argument('--num_heads', default=8, type=int, help=None)
    parser.add_argument('--mlp_ratios', default=4, type=int, help=None)
    parser.add_argument('--depths', default=4, type=int, help=None)
    parser.add_argument("--T", default=1, type=int, help=None)

    parser.add_argument("--batchsize", default=32, type=int, help=None)
    parser.add_argument("--num_workers", default=4, type=int, help=None)
    parser.add_argument('--dataset', default='CIFAR10', type=str, help=None)

    parser.add_argument("--boundary_angle", default=90, type=int, help=None)
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
    # aligned CLIP
    clip = torch.load(args.clip_path).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_name)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    clip_embedding = copy.deepcopy(clip.text_model.embeddings)

    # classes & templates
    module_name = f"label_sentence.{args.dataset}"
    module = importlib.import_module(module_name)
    classes = module.classes
    templates = module.templates

    # dataset
    train_dataset, test_dataset = my_load_dataset(args.dataset, image_processor, image_processor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False)

    # vision
    img_model = vision_encoder(
        in_channels=args.in_channels,
        target_dim=args.clip_dim,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        mlp_ratios=args.mlp_ratios,
        depths=args.depths,
        T=args.T
    ).to(args.device)
    checkpoint = torch.load(args.vision_path)
    print(checkpoint.keys()) 
    img_model.load_state_dict(checkpoint['state_dict'])

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

    # preparation for train
    optimizer = optim.Adam(img_model.parameters(), lr=args.lr, betas=args.betas, eps=1e-08, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    best_top1=0.
    best_epoch=0
    for epoch in tqdm(range(args.epoches)):
        train_losses = {"ce_loss":0., "kl_loss":0.}
        img_model.train()
        no_updata_num = 0
        for batch_idx, (images, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                images = images['pixel_values'][0].to(args.device)
                target = target.to(args.device)

                # clip_model
                clip_features = clip.get_image_features(images)
                clip_features = F.normalize(clip_features, p=2, dim=-1)
                clip_features = clip_features.detach()
                clip_features.requires_grad = False

                # img_model
                resized_img = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False) 
                image_features = img_model(resized_img)     
                image_features = F.normalize(image_features, p=2, dim=-1) 
                logits = image_features @ labels_candidate
                ce_loss = nn.CrossEntropyLoss()(logits, target)
                logits=F.log_softmax(logits, dim=-1)

                logits2 = clip_features @ labels_candidate
                logits2 = logits2.detach()
                logits2.requires_grad = False
                logits2=F.softmax(logits2,dim=-1)
                kl_loss = F.kl_div(logits, logits2, reduction='sum')
                train_losses["ce_loss"] += ce_loss
                train_losses["kl_loss"] += kl_loss     
            
            ce_grads = grad(ce_loss, img_model.parameters(), retain_graph=True)
            kl_grads = grad(kl_loss, img_model.parameters(), retain_graph=True)
        
            ce_tensor = torch.cat([tensor.flatten() for tensor in ce_grads], dim=0)
            kl_tensor = torch.cat([tensor.flatten() for tensor in kl_grads], dim=0)
            angle = calculate_angle(ce_tensor, kl_tensor)
            
            if args.boundary_angle > 0:
                if angle<=args.boundary_angle:
                    scaler.scale(ce_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    ce_loss.backward()
                    optimizer.zero_grad()
                    no_updata_num += 1
            else:
                scaler.scale(ce_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            functional.reset_net(img_model)

        scheduler.step()

        img_model.eval()
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images = images['pixel_values'][0].to(args.device)
                target = target.to(args.device)
                
                # predict
                resized_img = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False) # resized_img = images
                image_features = img_model(resized_img)
                image_features = F.normalize(image_features, p=2, dim=-1)
                logits = image_features @ labels_candidate

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
                functional.reset_net(img_model)
                # break

            # optput ACC
            top1 = (top1 / n) * 100
            top5 = (top5 / n) * 100 
            if top1>best_top1:
                best_top1=top1
                best_epoch=epoch
                torch.save(img_model, f'/data/ltlong/snnclip/output/{args.dataset}/step2_{top1}_{epoch}.pth')

        print(f"\n>>>>>  Epoch:[{epoch}/{args.epoches}]----> Stop:[{no_updata_num}/{len(train_loader)}], ce_loss:{train_losses['ce_loss']}, kl_loss:{train_losses['kl_loss']}, top1_acc:{top1:.4f}%, top5_acc:{top5:.4f}%, best_top1:{best_top1}, best_epoch:{best_epoch}\n")


if __name__ == '__main__':
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    
    set_seed(_args.seed)
    main(_args)




















