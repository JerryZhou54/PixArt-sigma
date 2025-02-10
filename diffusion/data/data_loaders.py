import os
import functools

import torch
import piat

def get_data_loader(data_dict, is_train=True, seed=0):
    image_size = data_dict['image_size']
    if image_size > 256:
        piat_filter_size = 1024
    elif image_size > 64:
        piat_filter_size = 256
    else:
        piat_filter_size = 64

    aesthetic_score = data_dict.get('aesthetic_score', 5.0)
    
    func_stages = [
        functools.partial(piat.filter_image, {'strCondition': f'(intWidth >= {piat_filter_size} or intHeight >= {piat_filter_size}) and fltAesthscore > {aesthetic_score}'}),
        functools.partial(piat.filter_text, {'strCondition': 'intLanguage == 2 and fltSimilarity >= 0.24'}), # only load samples with english text and a text similarity of 0.24 or more
        functools.partial(piat.image_load, {'strSource': f'{piat_filter_size}-pil-antialias'}),
        functools.partial(piat.image_alpha_smartskip, {}),
        functools.partial(piat.image_resize_antialias, {'intSize': image_size}),
        functools.partial(piat.image_crop_smart, {'intSize': image_size}),
        functools.partial(piat.text_load, {}),
        functools.partial(piat.output_image, {'fltMean': [0.5, 0.5, 0.5], 'fltStd': [0.5, 0.5, 0.5]}),
        functools.partial(piat.output_text, {}),
    ]

    # queryfile = 's3://sniklaus-clio-query/*/origin=l2&aesthetic=65p'
    queryfile = 's3://sniklaus-clio-query/2024.06.03/origin=l2&aesthetic=ours'

    def custom_collate_fn(batch):
        images = [item['tenImage'] for item in batch]
        texts = [item['strText'] for item in batch]
        
        # Stack the images into a single tensor and keep the texts as a list
        images_tensor = torch.stack(images, dim=0)
        return [images_tensor, texts]

    loader = piat.Dataloader(
        intBatchsize=data_dict['batch_size'],
        intWorkers=data_dict['workers'],
        intThreads=data_dict['threads'],
        strQueryfile=[queryfile],
        intSeed=0,
        funcStages=func_stages
    )

    return loader
