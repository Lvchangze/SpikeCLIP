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

from model import language_encoder, vision_encoder, CLIPv_encoder
from clip_eval import my_load_dataset, load_txt_dataset
from utils import *

set_seed(42)
clip_name = "openai/clip-vit-base-patch16"
clip = CLIPModel.from_pretrained(clip_name).cuda()
processor = CLIPProcessor.from_pretrained(clip_name)
tokenizer = processor.tokenizer
image_processor = processor.image_processor
clip_embedding = copy.deepcopy(clip.text_model.embeddings)
dataset_name = "CIFAR10"

module_name = f"label_sentence.{dataset_name}"
module = importlib.import_module(module_name)
classes = module.classes
templates = module.templates

train_dataset, test_dataset = my_load_dataset(dataset_name, image_processor, image_processor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=0, shuffle=False)

# #############################################################################################################
# ###################################         co-validation        ############################################
# #############################################################################################################
img_model = vision_encoder(
    in_channels=3,
    target_dim=512,
    embed_dims=384,
    num_heads=8,
    mlp_ratios=4,
    depths=4,
    T=4
).cuda()
vision_path = "vision_encoder_path"
img_model = CLIPv_encoder(
    in_channels=3,
    target_dim=512,
    embed_dims=384,
    num_heads=8,
    mlp_ratios=4,
    depths=4,
).cuda()
checkpoint = torch.load(vision_path)  
img_model.load_state_dict(checkpoint['state_dict'])
img_model.eval()

txt_model = language_encoder(
    pretrained=False,
    clip_embedding=clip_embedding,
    clip_dim=512,
    mlp_ratios=4,
    depths=4,
    T=4,
).cuda()
language_path = "/text_encoder_path/model_best.pth.tar"
checkpoint = torch.load(language_path)
txt_model.load_state_dict(checkpoint['state_dict'])
txt_model.eval()
labels_candidate = labels_candidates(classes, templates, txt_model, tokenizer).cuda()
print(f"labels_candidate = {labels_candidate.shape}")

with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for i, (images, target) in enumerate(tqdm(test_loader)):
        images = images['pixel_values'][0].cuda()
        target = target.cuda()
        
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
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

