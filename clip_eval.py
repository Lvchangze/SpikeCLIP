import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import torchvision
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import datasets
import importlib
import pickle
import torch.nn.functional as F


__all__ = ['my_load_dataset', 'load_txt_dataset']



def my_load_dataset(dataset_name, transform_train, transform_test, download=False, data_dir=None):
    if data_dir is None:
        data_dir = "/your_path/SpikeCLIP/datset/" + dataset_name
    
    torch.manual_seed(42)
    trainset, testset = None, None
    train_ratio = 0.8
    test_ratio = 0.2
    
    if dataset_name == "MNIST":
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=download, transform=transform_train)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=download, transform=transform_test)
        
    elif dataset_name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform_test)

    elif dataset_name == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transform_test)
    
    elif dataset_name == "ImageNet_1k":
        trainset = torchvision.datasets.ImageNet(root=data_dir, split='train', transform=transform_train)
        testset = torchvision.datasets.ImageNet(root=data_dir, split='val', transform=transform_train)
   
    elif dataset_name == "Food101":
        trainset = torchvision.datasets.Food101(root=data_dir, split='train', transform=transform_train, target_transform=None, download=download)
        testset = torchvision.datasets.Food101(root=data_dir, split='test', transform=transform_train, target_transform=None, download=download)
    
    elif dataset_name == "Flowers102":
        # trainset = torchvision.datasets.Flowers102(root=data_dir, split='train', transform=transform_train, target_transform=None, download=download)
        # testset = torchvision.datasets.Flowers102(root=data_dir, split='test', transform=transform_train, target_transform=None, download=download)
        testset = torchvision.datasets.Flowers102(root=data_dir, split='train', transform=transform_train, target_transform=None, download=download)
        trainset = torchvision.datasets.Flowers102(root=data_dir, split='test', transform=transform_train, target_transform=None, download=download)
    
    elif dataset_name == "OxfordIIITPet":
        trainset = torchvision.datasets.OxfordIIITPet(root=data_dir, split='trainval', target_types='category', transform=transform_train, target_transform=None, download=download)
        testset = torchvision.datasets.OxfordIIITPet(root=data_dir, split='test', target_types='category', transform=transform_train, target_transform=None, download=download)
    
    elif dataset_name == "STL10":
        trainset = torchvision.datasets.STL10(root=data_dir, split='train', folds=None, transform=transform_train, target_transform=None, download=download)
        testset = torchvision.datasets.STL10(root=data_dir, split='test', folds=None, transform=transform_train, target_transform=None, download=download)
    
    elif dataset_name == "SUN397":
        trainset = torchvision.datasets.SUN397(root=data_dir, transform=transform_train, target_transform=None, download=download)
        dataset_size = len(trainset)
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        trainset, testset = data.random_split(trainset, [train_size, test_size])
    
    elif dataset_name == "Caltech101":
        trainset = ImageFolder(f'{data_dir}/101_ObjectCategories', transform=transform_train)
        dataset_size = len(trainset)
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        trainset, testset = data.random_split(trainset, [train_size, test_size])
    
    elif dataset_name == "tiny_imagenet":
        dataset = load_dataset("zh-plus/tiny-imagenet")
        trainset, testset = dataset['train'], dataset['valid']

       
    return trainset, testset


def labels_candidates(classnames, templates, model, tokenizer):
    device = next(model.parameters()).device
    with torch.no_grad():
        candidates_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            y = tokenizer(text=texts, return_tensors="pt", padding=True) #tokenize
            y = {key: value.to(device) for key, value in y.items()}

            class_embeddings = model.get_text_features(y['input_ids'], y['attention_mask']) #embed with text encoder
            # class_embeddings = model(y['input_ids'], y['attention_mask']) #embed with text encoder
            
            class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embedding, p=2, dim=-1)
            candidates_weights.append(class_embedding)
        candidates_weights = torch.stack(candidates_weights, dim=1)
    return candidates_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def eval_clip(dataset_name):
    # load classes&template
    module_name = f"label_sentence.{dataset_name}"
    module = importlib.import_module(module_name)
    classes = module.classes
    templates = module.templates
    print(f">>>>>>>>>>   Now, peformer evaluation on {dataset_name}   <<<<<<<<<<")
    print(f"{len(classes)} classes, {len(templates)} templates")
    
    # load CLIP
    clip_name = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(clip_name).cuda()
    preprocess = CLIPProcessor.from_pretrained(clip_name)
    text_config = model.config.text_config
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print(f"text_config:{text_config}")
    
    # load dataset
    train_dataset, test_dataset = my_load_dataset(dataset_name, preprocess.image_processor, preprocess.image_processor)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=2)
    
    # make labels tensors
    zeroshot_weights = labels_candidates(classes, templates, model, preprocess)
    print(f"zeroshot_weights = {zeroshot_weights.shape}")       
    
    # do evaluation
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images['pixel_values'][0].cuda()
            target = target.cuda()
            
            # predict
            image_features = model.get_image_features(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
            logits = image_features @ zeroshot_weights
            
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            # break
    # optput ACC
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
  
    
def make_txt_dataset(dataset_list, save_path):
    all_texts = []
    
    for dataset_name in dataset_list:
        module_name = f"label_sentence.{dataset_name}"
        module = importlib.import_module(module_name)
        classes = module.classes
        templates = module.templates
        
        for classname in tqdm(classes):
            texts = [template.format(classname) for template in templates]
            all_texts += texts
    
    with open(save_path, 'w') as f:  
        for text in all_texts:
            f.write(text + '\n')
    

def load_txt_dataset(save_path='/your_path/SpikeCLIP/language/label_sentence/label_text.txt'):
    all_texts = []

    with open(save_path, 'r') as f: 
        for line in f:
            text = line.strip()  
            all_texts.append(text)

    return all_texts

