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
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from ..utils import *
from ..model import language_encoder, vision_encoder, CLIPv_encoder
from ..clip_eval import my_load_dataset, load_txt_dataset



def args():
    parser = argparse.ArgumentParser(description='Dual-Loss Finetuning', add_help=False)
    
    parser.add_argument('--device', default="cuda:0", type=str, help=None)
    parser.add_argument('--seed', default=42, type=int, help=None)
    parser.add_argument('--clip_name', default="openai/clip-vit-base-patch16", type=str, help=None)
    parser.add_argument('--clip_path', default=None, type=str, help=None)
    parser.add_argument('--clip_dim', default=512, type=int, help=None)
    parser.add_argument('--embed_dims', default=384, type=int, help=None)
    parser.add_argument('--in_channels', default=3, type=int, help=None)
    parser.add_argument('--num_heads', default=8, type=int, help=None)
    parser.add_argument('--mlp_ratios', default=4, type=int, help=None)
    parser.add_argument('--depths', default=4, type=int, help=None)
    parser.add_argument("--T", default=4, type=int, help=None)
    parser.add_argument('--vision_path', default=None, type=str, help=None)
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
    parser.add_argument('--boundary_angle', default=90, type=int, help=None)

    parser.add_argument('-g', '--gpuid', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local-rank", type=int, help='rank in current node')                                 
    parser.add_argument('--use_mix_precision', default=True, action='store_true', help="whether to use mix precision") 

    args = parser.parse_args()
    return args


def main(args):
    gpu=args.local_rank

    # classes & templates
    module_name = f"label_sentence.{args.dataset}"
    module = importlib.import_module(module_name)
    classes = module.classes
    templates = module.templates
    print(f">>>>>>>>>>   The evaluation  will be performe on {args.dataset}   <<<<<<<<<<")
    print(f"{len(classes)} classes, {len(templates)} templates")

    dist.init_process_group(backend='nccl', init_method='env://')    
    args.rank = dist.get_rank()                                      
    
    # aligned CLIP
    clip_model = torch.load(args.clip_path)
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
    )
    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    checkpoint = torch.load(args.vision_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda(gpu)
    clip_model.cuda(gpu)

    # text
    txt_model = language_encoder(
        pretrained=False,
        clip_embedding=clip_embedding,
        clip_dim=args.clip_dim,
        mlp_ratios=args.mlp_ratios,
        depths=args.depths,
        T=args.T
    ).cuda(gpu)
    checkpoint = torch.load(args.language_path)
    txt_model.load_state_dict(checkpoint['state_dict'])
    labels_candidate = labels_candidates(classes, templates, txt_model, tokenizer).to(gpu)
    labels_candidate.requires_grad = False

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)                  
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])   
    scaler = torch.cuda.amp.GradScaler()                  
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=1e-08, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
     
    if args.rank == 0:
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

    # dataset
    trainset, testset =my_load_dataset(dataset_name=args.dataset, transform_train=img_processor, transform_test=img_processor)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset)   

    train_loader = DataLoader(
        trainset,
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=train_sampler
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,  
        sampler=test_sampler
    ) 
   
    epoches=args.epoches
    batchsize=args.batchsize
    start = datetime.now()
    for epoch in tqdm(range(epoches)):
        # #############################################################################################################
        # ###################################             Train            ############################################
        # #############################################################################################################
        train_loader.sampler.set_epoch(epoch)    
        model.train()
        train_losses = {"ce_loss":0., "kl_loss":0.}
        no_updata_num = 0

        for batch_idx, (images, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                target = target.to(gpu)
                images = images['pixel_values'][0].to(gpu)
                resized_img = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False) 
                model_features = model(resized_img)
                clip_features = clip_model.get_image_features(images)
                clip_features = F.normalize(clip_features, p=2, dim=-1)
                clip_features = clip_features.detach()
                clip_features.requires_grad = False
                image_features = F.normalize(model_features, p=2, dim=-1)
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
    
            ce_grads = grad(ce_loss, model.parameters(), retain_graph=True)
            kl_grads = grad(kl_loss, model.parameters(), retain_graph=True)
        
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
                scaler.scale(ce_loss+kl_loss).backward()
                scaler.step(optimizer)
                scaler.update()  
            functional.reset_net(model)
            
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        # #############################################################################################################
        # ###################################           Validation         ############################################
        # #############################################################################################################
        model.eval()
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(test_loader)):
                images = images['pixel_values'][0].to(gpu)
                target = target.to(gpu)
                
                # predict
                resized_img = F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
                image_features = model(resized_img)
                image_features = F.normalize(image_features, p=2, dim=-1)
                logits = image_features @ labels_candidate

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
                functional.reset_net(model)
                # break

            # optput ACC
            top1 = torch.tensor(top1).to(gpu)
            top5 = torch.tensor(top5).to(gpu)
            n = torch.tensor(n).to(gpu)
            dist.reduce(top1, 0, op=dist.ReduceOp.SUM)
            dist.reduce(top5, 0, op=dist.ReduceOp.SUM)
            dist.reduce(n, 0, op=dist.ReduceOp.SUM)

            top1 = (top1 / n) * 100
            top5 = (top5 / n) * 100 

        if args.rank == 0:
            best_top1, best_epoch = saver.save_checkpoint(epoch, metric=top1)
            # print(f"\n>>>>>  Epoch:[{epoch+1}/{epoches}]----> train_loss:{train_losses}, lr:{current_lr:.2e}, top1_acc:{top1:.2f}%, top5_acc:{top5:.2f}%, best_top1:{best_top1:.2f}%, best_epoch:{best_epoch}")
            print(f"\n>>>>>  Epoch:[{epoch}/{args.epoches}]----> current_lr:{current_lr:.2e}, Stop:[{no_updata_num}/{len(train_loader)}], ce_loss:{train_losses['ce_loss']}, kl_loss:{train_losses['kl_loss']}, top1_acc:{top1:.4f}%, top5_acc:{top5:.4f}%, best_top1:{best_top1}, best_epoch:{best_epoch}\n")
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Current time: {time}\n")
            
    dist.destroy_process_group()    
    if args.rank == 0:              
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    
    set_seed(_args.seed)
    main(_args)













    
    
    
    