import logging
import json
import torch
from types import SimpleNamespace
from vilbert.vilbert import VILBertActionGrounding, BertConfig
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch.distributed as dist
from VLN_config import config as args
import random
import pandas as pd
from dataLoader import DataLoader

import sys
import os
import torch
import yaml

import numpy as np
import matplotlib.pyplot as plt
import PIL

from PIL import Image
import cv2
import argparse
import glob
import pdb

import torchvision.models as models
import torchvision.transforms as transforms

from faster_rcnn import feature_extractor_new as f_extractor
from faster_rcnn.feature_extractor_new import featureExtractor

def split_train_val (data_loaded, split_portion = 0.9):
    ''' Function to prepare splits for data Loaded
    WE are going to ignore infos
    '''
    features_masked, pos_enc, spatial, image_mask, tokenized_text, masked_text, masked_lm_token, input_mask, segment_ids, co_attention_mask, infos, masked_img_labels  = data_loaded
    
    indexes = list(range(features_masked.shape[0]))
    random.shuffle(indexes)
    split_train = int(features_masked.shape[0] * split_portion)
    indexes_train = indexes[:split_train]
    indexes_val = indexes[split_train:]
    data_loaded_train = (features_masked[indexes_train], 
                         pos_enc[indexes_train], 
                         spatial[indexes_train], 
                         image_mask[indexes_train], 
                         tokenized_text[indexes_train], 
                         masked_text[indexes_train], 
                         masked_lm_token[indexes_train], 
                         input_mask[indexes_train], 
                         segment_ids[indexes_train], 
                         co_attention_mask[indexes_train],  
                         masked_img_labels[indexes_train])
    data_loaded_val = (features_masked[indexes_val], 
                         pos_enc[indexes_val], 
                         spatial[indexes_val], 
                         image_mask[indexes_val], 
                         tokenized_text[indexes_val], 
                         masked_text[indexes_val], 
                         masked_lm_token[indexes_val], 
                         input_mask[indexes_val], 
                         segment_ids[indexes_val], 
                         co_attention_mask[indexes_val],  
                         masked_img_labels[indexes_val])
    return data_loaded_train, data_loaded_val 

# load data
frcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
data_loader = DataLoader("json_data.json", frcnn_model, save_or_not = False)
path = 'DataLoader.pt'
data_loaded = data_loader.load_dataloader(path)
print('data Loaded successfully !')
# Split the data
data_train , data_val = split_train_val(data_loaded)
(features_masked_train, pos_enc_train, spatial_train, image_mask_train, tokenized_text_train, masked_text_train,
     masked_lm_token_train, input_mask_train, 
    segment_ids_train, co_attention_mask_train, masked_img_labels_train)  = data_train

(features_masked_val, pos_enc_val, spatial_val, image_mask_val, tokenized_text_val, masked_text_val,
     masked_lm_token_val, input_mask_val, 
    segment_ids_val, co_attention_mask_val, masked_img_labels_val)  = data_val

# Call Vilbert
config = BertConfig.from_json_file(args.config_file)
bert_weight_name = json.load(
    open("config/" + args.bert_model + "_weight_name.json", "r")
)

tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=args.do_lower_case
)

config.track_temporal_features = args.track_temporal_features
config.mean_layer = args.mean_layer
config.max_temporal_memory_buffer = args.max_temporal_memory_buffer

model = VILBertActionGrounding.from_pretrained(
    args.from_pretrained, config=config, default_gpu=True
)
for key, value in dict(model.named_parameters()).items():
    if not value.requires_grad:
        print("This parameter does have grad", key)
print('Vilbert Loaded successfully !')

# Start Action Grounding
model.cuda()
model.train()
optimizer = AdamW(model.parameters(),
                    lr=args.learning_rate,
                    eps=args.adam_epsilon,
                    betas=(0.9, 0.98),)

batch_size = args.train_batch_size 
loss_result_csv = pd.DataFrame(columns = ['epochs', 'train_loss', 'val_loss'])

for epoch in range(100):
    i = 0
    loss_train_cum = 0
    while (i < data_train[0].shape[0]):
        pred_t_train, pred_v_train, att_train = model(input_ids = masked_text_train[i:i+batch_size].cuda(),
                                image_feat = features_masked_train[i:i+batch_size].cuda(), # Linear(2048*config.max_temporal_memory_buffer, 2048)
                                image_loc = spatial_train[i:i+batch_size].cuda(),  #Linear(in_features=5, out_features=1024, bias=True)
                                image_pos_input = pos_enc_train[i:i+batch_size].cuda(),   #Linear(7, 2048)/(6, 2048)
                                token_type_ids = segment_ids_train[i:i+batch_size].cuda(), 
                                attention_mask = input_mask_train[i:i+batch_size].cuda(), 
                                image_attention_mask = image_mask_train[i:i+batch_size].cuda(),
                                output_all_attention_masks=True)
        i = i+batch_size
        optimizer.zero_grad()
        masked_lm_loss_train = model.lang_criterion(pred_t_train.view(-1, 30522), masked_lm_token_train.cuda().view(-1))
        img_loss_train = model.vis_criterion(pred_v_train.view(-1, 91), masked_img_labels_train.view(-1).cuda()) # why dim 2 (to check) 
        loss_train = masked_lm_loss_train + img_loss_train
        loss_train.backward()
        loss_train_cum += loss_train
        optimizer.step()
    loss_train_cum = loss_train_cum/data_train[0].shape[0]
    print("epoch: " , epoch, " Train loss: ", loss_train_cum)
    # Validation
    pred_t_val, pred_v_val, att_val = model(input_ids = masked_text_val.cuda(),
                            image_feat = features_masked_val.cuda(), # Linear(2048*config.max_temporal_memory_buffer, 2048)
                            image_loc = spatial_val.cuda(),  #Linear(in_features=5, out_features=1024, bias=True)
                            image_pos_input = pos_enc_val.cuda(),   #Linear(7, 2048)/(6, 2048)
                            token_type_ids = segment_ids_val.cuda(), 
                            attention_mask = input_mask_val.cuda(), 
                            image_attention_mask = image_mask_val.cuda(),
                            output_all_attention_masks=True)
    optimizer.zero_grad()
    masked_lm_loss_val = model.lang_criterion(pred_t_val.view(-1, 30522), masked_lm_token_val.cuda().view(-1))
    img_loss_val = model.vis_criterion(pred_v_val.view(-1, 91), masked_img_labels_val.view(-1).cuda()) # why dim 2 (to check) 
    loss_val = masked_lm_loss_val + img_loss_val
    loss_val = loss_val / data_val[0].shape[0]
    print("epoch: " , epoch," Val loss: ", loss_val)
    loss_result_csv = loss_result_csv.append(pd.DataFrame(epoch, loss_train_cum, loss_val))
    if i % 10 == 0:
        loss_result_csv.to_csv('epoch_loss.csv')
        model.save_pretrained('vilbert_action_grounding.bin')

