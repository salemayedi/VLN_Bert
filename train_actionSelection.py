import logging
import json
import torch
from types import SimpleNamespace
from vilbert.vilbert import VILBertActionSelection, BertConfig
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch.distributed as dist
from VLN_config import config as args
from torch.nn import CrossEntropyLoss
import random
import pandas as pd
from data.dataLoaderActSelection import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
# %matplotlib inline


def split_train_val(data_loaded, split_portion=0.9):
    # WE are going to ignore infos
    features, pos_enc, spatial, image_mask, tokenized_text, input_mask, segment_ids, co_attention_mask, infos, action_targets = data_loaded

    indexes = list(range(features.shape[0]))
    random.shuffle(indexes)
    split_train = int(features.shape[0] * split_portion)
    indexes_train = indexes[:split_train]
    indexes_val = indexes[split_train:]
    data_loaded_train = (features[indexes_train],
                         pos_enc[indexes_train],
                         spatial[indexes_train],
                         image_mask[indexes_train],
                         tokenized_text[indexes_train],
                         input_mask[indexes_train],
                         segment_ids[indexes_train],
                         co_attention_mask[indexes_train],
                         action_targets[indexes_train])
    data_loaded_val = (features[indexes_val],
                       pos_enc[indexes_val],
                       spatial[indexes_val],
                       image_mask[indexes_val],
                       tokenized_text[indexes_val],
                       input_mask[indexes_val],
                       segment_ids[indexes_val],
                       co_attention_mask[indexes_val],
                       action_targets[indexes_val])
    return data_loaded_train, data_loaded_val


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# load data
frcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
data_loader = DataLoader("data/json_data.json", frcnn_model, save_or_not=False)
path = 'data/DataLoaderActSelection.pt'
data_loaded = data_loader.load_dataloader(path)
print('data Loaded successfully !')

data_train, data_val = split_train_val(data_loaded)

(features_train, pos_enc_train, spatial_train, image_mask_train, tokenized_text_train, input_mask_train,
    segment_ids_train, co_attention_mask_train, action_targets_train) = data_train

(features_val, pos_enc_val, spatial_val, image_mask_val, tokenized_text_val, input_mask_val,
    segment_ids_val, co_attention_mask_val, action_targets_val) = data_val
print('data Splitted successfully !')

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
config.visualization = True

model = VILBertActionSelection.from_pretrained(
    args.from_pretrained, config=config, default_gpu=True
)
# Check if all parameters model have gradients activated
for key, value in dict(model.named_parameters()).items():
    if not value.requires_grad:
        print("This parameter does have grad", key)
print('Vilbert Loaded successfully !')


# Set the model for training
print("Exist Cuda: ", torch.cuda.is_available())
print(torch.cuda.get_device_name())
model.cpu()
model.train()
# Change adamW for action selection
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
# optimizer = AdamW(model.parameters(),
#                   lr=args.learning_rate,
#                   eps=args.adam_epsilon,
#                   betas=(0.9, 0.98),)
criterion = CrossEntropyLoss()

batch_size = args.train_batch_size
if args.use_tensorboard:
    writer = SummaryWriter()
best_train = 100000
best_val = 1000000
print("\n START !   \n")
for epoch in range(args.epochs):
    i = 0
    loss_train_cum = 0.
    num_batches = data_train[0].shape[0]//batch_size+1
    #import pdb; pdb.set_trace()
    for i in range(num_batches):
        if (i == num_batches - 1):
            #print("last elements of the Batch!")
            r = data_train[0].shape[0] % batch_size
            pred_action_train, att_train = model(input_ids=tokenized_text_train[data_train[0].shape[0]-batch_size:].cpu(),
                                                 # Linear(2048*config.max_temporal_memory_buffer, 2048)
                                                 image_feat=features_train[data_train[0].shape[0]-batch_size:].cpu(),
                                                 # Linear(in_features=5, out_features=1024, bias=True)
                                                 image_loc=spatial_train[data_train[0].shape[0]-batch_size:].cpu(),
                                                 # Linear(7, 2048)/(6, 2048)
                                                 image_pos_input=pos_enc_train[data_train[0].shape[0] - \
                                                                               batch_size:].cpu(),
                                                 token_type_ids=segment_ids_train[data_train[0].shape[0] - \
                                                                                  batch_size:].cpu(),
                                                 attention_mask=input_mask_train[data_train[0].shape[0] - \
                                                                                 batch_size:].cpu(),
                                                 image_attention_mask=image_mask_train[data_train[0].shape[0] - \
                                                                                       batch_size:].cpu(),
                                                 output_all_attention_masks=True)
            # Check the shapes in the criterion
            action_target_batch = action_targets_train[data_train[0].shape[0]-batch_size:].view(-1).cpu()
            #print(action_target_batch.shape, pred_action_train.shape)
            loss_train = criterion(pred_action_train, action_target_batch)
        else:
            pred_action_train, att_train = model(input_ids=tokenized_text_train[i*batch_size:(i+1)*batch_size].cpu(),
                                                 # Linear(2048*config.max_temporal_memory_buffer, 2048)
                                                 image_feat=features_train[i*batch_size:(i+1)*batch_size].cpu(),
                                                 # Linear(in_features=5, out_features=1024, bias=True)
                                                 image_loc=spatial_train[i*batch_size:(i+1)*batch_size].cpu(),
                                                 # Linear(7, 2048)/(6, 2048)
                                                 image_pos_input=pos_enc_train[i*batch_size:(i+1)*batch_size].cpu(),
                                                 token_type_ids=segment_ids_train[i*batch_size:(i+1)*batch_size].cpu(),
                                                 attention_mask=input_mask_train[i*batch_size:(i+1)*batch_size].cpu(),
                                                 image_attention_mask=image_mask_train[i * \
                                                                                       batch_size:(i+1)*batch_size].cpu(),
                                                 output_all_attention_masks=True)
            # Check the shapes in the criterion
            action_target_batch = action_targets_train[i*batch_size:(i+1)*batch_size].view(-1).cpu()
            #print(action_target_batch.shape, pred_action_train.shape)

            loss_train = criterion(pred_action_train, action_target_batch)
            optimizer.zero_grad()
        loss_train.backward()
        loss_train_cum += loss_train
        optimizer.step()
    loss_train_cum = loss_train_cum/data_train[0].shape[0]
    # Validation
    pred_action_val, att_val = model(input_ids=tokenized_text_val.cpu(),
                                     image_feat=features_val.cpu(),  # Linear(2048*config.max_temporal_memory_buffer, 2048)
                                     image_loc=spatial_val.cpu(),  # Linear(in_features=5, out_features=1024, bias=True)
                                     image_pos_input=pos_enc_val.cpu(),  # Linear(7, 2048)/(6, 2048)
                                     token_type_ids=segment_ids_val.cpu(),
                                     attention_mask=input_mask_val.cpu(),
                                     image_attention_mask=image_mask_val.cpu(),
                                     output_all_attention_masks=True)
    #import pdb; pdb.set_trace()
    optimizer.zero_grad()
    loss_val = criterion(pred_action_val, action_targets_val.view(-1).cpu())
    loss_val = loss_val / data_val[0].shape[0]
    print("epoch: ", epoch, "Train loss: ", loss_train_cum.item(), " Val loss: ", loss_val.item())
    if args.use_tensorboard:
        # Plot separately the losses img and lm
        writer.add_scalar('Loss/train', loss_train_cum, epoch)
        writer.add_scalar('Loss/validation', loss_val, epoch)
    #loss_result_csv = loss_result_csv.append(pd.DataFrame([[epoch, loss_train_cum.item(), loss_val.item()]], columns = loss_result_csv.columns ), ignore_index = True)
    if best_val > loss_val.item():
        best_val = loss_val.item()
        torch.save(model.state_dict(), "save_vilbert_action_selection/best_val_vilberActionSelection.bin")
        print("Model saved best validation !")
    if best_train > loss_train_cum.item():
        best_train = loss_train_cum.item()
        torch.save(model.state_dict(), "save_vilbert_action_selection/best_train_vilberActionSelection.bin")
        print("Model saved best Train !")

writer.close()
