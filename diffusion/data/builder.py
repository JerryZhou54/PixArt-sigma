import os
import time

from mmcv import Registry, build_from_cfg
from torch.utils.data import DataLoader

from diffusion.data.transforms import get_transform
from diffusion.utils.logger import get_root_logger

DATASETS = Registry('datasets')

DATA_ROOT = '/cache/data'


def set_data_root(data_root):
    global DATA_ROOT
    DATA_ROOT = data_root


def get_data_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return os.path.join(DATA_ROOT, data_dir)


def get_data_root_and_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return DATA_ROOT, os.path.join(DATA_ROOT, data_dir)


def build_dataset(cfg, resolution=224, **kwargs):
    logger = get_root_logger()

    dataset_type = cfg.get('type')
    logger.info(f"Constructing dataset {dataset_type}...")
    t = time.time()
    transform = cfg.pop('transform', 'default_train')
    transform = get_transform(transform, resolution)
    dataset = build_from_cfg(cfg, DATASETS, default_args=dict(transform=transform, resolution=resolution, **kwargs))
    logger.info(f"Dataset {dataset_type} constructed. time: {(time.time() - t):.2f} s, length (use/ori): {len(dataset)}/{dataset.ori_imgs_nums}")
    return dataset


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    if 'batch_sampler' in kwargs:
        dataloader = DataLoader(dataset, batch_sampler=kwargs['batch_sampler'], num_workers=num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True,
                                **kwargs)
    return dataloader

from torch.utils.data import Dataset
import datasets as ds
class CoCoCaption(Dataset): 
    def __init__(self, transform): 
        self.dataset = ds.load_dataset("shunk031/MSCOCO", "2014-captions", split='train')
        self.transform = transform

    def __len__(self):
        return len(self.dataset) * 5

    def __getitem__(self, index): 
        idx = index // 5
        offset = index % 5
        image = self.transform(self.dataset[idx]['image'].convert("RGB"))
        caption = self.dataset[idx]['annotations']['caption'][offset]
        return image, caption

import json
from PIL import Image
class LaionDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        with open(os.path.join(os.path.dirname(path), "index.json"), "r") as f:
            self.idx = json.load(f)
    
    def __len__(self):
        jpg_count = sum(1 for file in os.listdir(self.path) if file.lower().endswith('.jpg'))
        return jpg_count
    
    def __getitem__(self, index):
        idx = self.idx[str(index)]
        img_path = os.path.join(self.path, f"{idx:09}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        json_path = os.path.join(self.path, f"{idx:09}.json")
        with open(json_path, "r") as f:
            stats = json.load(f)
            txt = stats["caption"]
        return img, txt