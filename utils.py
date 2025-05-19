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

from clip_eval import my_load_dataset, load_txt_dataset








def cosine_similarity_loss(output, clip_features):
    norm1 = torch.norm(output, dim=1, keepdim=True)
    norm2 = torch.norm(clip_features, dim=1, keepdim=True)

    norm1 = torch.max(norm1, torch.tensor([1e-8], device=norm1.device))
    norm2 = torch.max(norm2, torch.tensor([1e-8], device=norm2.device))
    
    output = output / norm1
    clip_features = clip_features / norm2
    similarity = F.cosine_similarity(output, clip_features.to(output.device), dim=1)
    distance = 1 - similarity
    loss = torch.mean(distance)
    return loss


def labels_candidates(classnames, templates, model, tokenizer):
    device = next(model.parameters()).device
    with torch.no_grad():
        candidates_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            y = tokenizer(text=texts, return_tensors="pt", max_length=20, padding="max_length")
            y = {key: value.to(device) for key, value in y.items()}

            # class_embeddings = model.get_text_features(y['input_ids'], y['attention_mask']) #embed with text encoder
            class_embeddings = model(y['input_ids'], y['attention_mask']) #embed with text encoder
            
            class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
            class_embedding = class_embeddings.mean(dim=0)
            candidates_weights.append(class_embedding)
        candidates_weights = torch.stack(candidates_weights, dim=1)
    return candidates_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def compute_similarity_scores(image_features, zeroshot_weights):
    B, D = image_features.shape
    D, N = zeroshot_weights.shape
    
    image_features = image_features.unsqueeze(1)
    zeroshot_weights = zeroshot_weights.transpose(0, 1).unsqueeze(0)
    
    similarities = F.cosine_similarity(image_features, zeroshot_weights, dim=2)
    similarity_scores = similarities.view(B, N)
    similarity_scores = (similarity_scores + 1)/2.
    
    return similarity_scores


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_angle(tensor1, tensor2):
    dot_product = torch.dot(tensor1, tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    cosine_similarity = dot_product / (norm1 * norm2)
    angle_rad = torch.acos(cosine_similarity)
    angle_deg = angle_rad * (180.0 / 3.141592653589793)
    return angle_deg.item()






