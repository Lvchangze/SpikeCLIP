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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from transformers import CLIPProcessor, CLIPModel
from spikingjelly.clock_driven import functional
from timm.utils import get_outdir, CheckpointSaver
from model import language_encoder
from clip_eval import my_load_dataset, load_txt_dataset

from ..utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/text')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def args():
    parser = argparse.ArgumentParser(description='Training Language-Encoder Config', add_help=False)
    
    parser.add_argument('--device', default="cuda:0", type=str, help=None)
    parser.add_argument('--seed', default=42, type=int, help=None)
    parser.add_argument('--clip_name', default="openai/clip-vit-base-patch16", type=str, help=None)
    parser.add_argument('--embed_dims', default=384, type=int, help=None)
    parser.add_argument('--num_heads', default=8, type=int, help=None)
    parser.add_argument('--mlp_ratios', default=4, type=int, help=None)
    parser.add_argument('--depths', default=4, type=int, help=None)
    parser.add_argument("--T", default=4, type=int, help=None)
    parser.add_argument('--checkpoint_path', default=None, type=str, help=None)
    parser.add_argument("--lr", default=5e-5, type=float, help=None)
    parser.add_argument("--betas", default=(0.9, 0.999), type=tuple, help=None)
    parser.add_argument("--weight_decay", default=0., type=float, help=None)
    parser.add_argument("--amsgrad", default=False, type=bool, help=None)
    parser.add_argument('--T_max', default=200, type=int, help=None)
    parser.add_argument('--eta_min', default=2e-8, type=float, help=None)
    parser.add_argument('--eval_metric', default='top1', type=str, help=None)
    parser.add_argument('--max_history', default=5, type=int, help=None)
    parser.add_argument('--eval_dataset_name', default='CIFAR10', type=str, help=None)
    parser.add_argument('--epoches', default=2000, type=int, help=None)
    parser.add_argument('--batch_size', default=512, type=int, help=None)

    args = parser.parse_args()
    return args


def train(
    epoches,
    model,
    tokenizer,
    clip_model,
    txt_list,
    batch_size,
    optimizer,
    loss_fn,
    lr_scheduler,

    output_dir=None,
    scaler=None,

    test_dataset=None,
    classes=None,
    templates=None,
    saver=None,
    
    args=None
):  
    best_top1 = 0.
    best_loss = 100.
    steps = 0
    patience = 0
    for epoch in tqdm(range(epoches)):

        # #########################################################################################################################
        # #########################################################################################################################
        model.train()
        num = len(txt_list)
        last_idx = num//batch_size - 1
        num_updates = epoch * (num//batch_size)
        train_losses = 0.

        index_list = torch.randperm(num)
        
        for batch_idx in tqdm(range(num//batch_size)):
            last_batch = batch_idx == last_idx
            idx = batch_idx*batch_size
            texts = [txt_list[i] for i in index_list[idx : idx+batch_size]]
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # language encoder
                input = tokenizer(text=texts, return_tensors="pt", max_length=20, padding="max_length")

                input_ids = input['input_ids'].to(args.device)
                attention_mask = input['attention_mask'].to(args.device)
                output = model(input_ids, attention_mask)   

                # clip 
                txt_features = clip_model.get_text_features(input_ids, attention_mask)
                
                # cosine_similarity
                loss = loss_fn(output, txt_features)
                train_losses += loss
                writer.add_scalars("txt_train_loss", {'txt_loss': loss}, steps)
                steps += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(model)
            # break


        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']



        # #########################################################################################################################
        # #########################################################################################################################
        model.eval()
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=4)
        zeroshot_weights = labels_candidates(classes, templates, model, tokenizer).to(args.device)
        
        # do evaluation
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images['pixel_values'][0].to(args.device)
                target = target.to(args.device)
                
                # predict
                image_features = clip_model.get_image_features(images)
                image_features = F.normalize(image_features, p=2, dim=-1)
                logits = image_features @ zeroshot_weights

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
                functional.reset_net(model)

        # optput ACC
        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100 
        
        if train_losses > best_loss:
            patience +=1
        else:
            patience = 0
        best_loss, best_epoch = saver.save_checkpoint(epoch, metric=train_losses)
        
        print(f"\n>>>>>  Epoch:[{epoch+1}/{epoches}]----> train_loss:{train_losses}, current_lr:{current_lr:.4e}, top1_acc:{top1:.4f}%, top5_acc:{top5:.4f}%, best_epoch:{best_epoch}, best_loss:{best_loss}\n")

        if patience >= 20:
            print(f">>>>>>  Training is over because patience  <<<<<<")
            break


def main(args):
    device=args.device
    clip = CLIPModel.from_pretrained(args.clip_name)
    processor = CLIPProcessor.from_pretrained(args.clip_name)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    clip_embedding = copy.deepcopy(clip.text_model.embeddings)
    model = language_encoder(
        pretrained=False,
        clip_embedding=clip_embedding,
        clip_dim=512,
        mlp_ratios=args.mlp_ratios,
        depths=args.depths,
        T=args.T,
    ).to(device)
    clip = clip.to(device)
    for param in clip.parameters():
        param.requires_grad = False
    txt_list = load_txt_dataset()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=1e-08, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"),])
    output_dir = get_outdir('./output/last_text', exp_name)
    eval_metric = args.eval_metric
    decreasing = True if eval_metric == 'loss' else False

    saver = CheckpointSaver(
        model=model,
        optimizer=optimizer,
        args=None,
        amp_scaler=None,
        checkpoint_dir=output_dir,
        recovery_dir=output_dir,
        decreasing=decreasing,
        max_history=args.max_history
    )

    eval_dataset_name = args.eval_dataset_name
    module_name = f"label_sentence.{eval_dataset_name}"
    module = importlib.import_module(module_name)
    classes = module.classes
    templates = module.templates
    train_dataset, test_dataset = my_load_dataset(eval_dataset_name, image_processor, image_processor)
    train(
        epoches=args.epoches,
        model=model,
        tokenizer=tokenizer,
        clip_model=clip,
        txt_list=txt_list,
        batch_size=args.batch_size,
        optimizer=optimizer,
        loss_fn=cosine_similarity_loss,
        lr_scheduler=scheduler,

        output_dir=output_dir,
        scaler=scaler,
        test_dataset=test_dataset,
        classes=classes,
        templates=templates,
        saver=saver,
        
        args=args
    )
    

if __name__ == '__main__':
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    
    set_seed(_args.seed)
    main(_args)






