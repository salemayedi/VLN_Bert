from copy import deepcopy
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


tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=args.do_lower_case
)


class DataLoader():
    """This class loads and preprocess the data set ALFRED for VilBERT.
        Input: data.json and model for the feature extractor (FastRCnn)
        Ouput: List of length = number of different instructions on our dataset. Each element of the list
                is a dictionary containing a sequence of images + instruction (text), under the keys
                [imgs] & [desc] respectively. [imgs] its a dictionary with keys [features],
                [pos_enco] and [infos], gathering the already masked and tokenized features extracted from
                the fasRCnn, a positional encoder of the bounding boxes and some additional information.
                [desc] is a dictionary with keys [text_id],[modified_token] and [masked_lm_token], gathering
                the tokenized instruction, the modification after making it and the masked_lm_token for VilBERT,
                respectively. """

    def __init__(self, json_path, token_count, model, save_or_not=False):
        self.data = json.load(open(json_path, "r"))[:120]
        self.dataset_token_count = json.load(open(token_count, "r"))
        print(len(self.data))
        # assert False
        self.tokenized_data = []
        self.model = model
        self.save_or_not = save_or_not

    def get_processed_data(self):
        return self.tokenized_data

    def get_top_used_words(self, p):
        top = self.dataset_token_count[:int(p*len(self.dataset_token_count))]
        top = [tokenizer.convert_tokens_to_ids(a["token"]) for a in top]
        return top

    def add_randomly_selected_box(self, features, positional_encoding, infos):
        i = np.random.randint(features.shape[0])
        features = torch.cat((features, features[i].reshape(1, -1)), dim=0)
        positional_encoding = torch.cat((positional_encoding, positional_encoding[i].reshape(1, -1)), dim=0)
        infos["bbox"] = np.concatenate((infos["bbox"], infos["bbox"][i].reshape(1, -1)), axis=0)
        infos["pos_enc"] = np.concatenate((infos["pos_enc"], infos["pos_enc"][i].reshape(1, -1)), axis=0)
        infos["objects"] = torch.cat((infos["objects"], infos["objects"][i].unsqueeze(0)), dim=0)
        infos["cls_prob"] = np.concatenate((infos["cls_prob"], infos["cls_prob"][i].reshape(1)), axis=0)
        infos["num_boxes"] += 1
        return features, positional_encoding, infos

    def flatten_to_one_img(self, features, positional_encoding, infos):
        """

        """
        features = torch.cat(features, dim=0)
        positional_encoding = torch.cat(positional_encoding, dim=0)
        infos_bbox = np.concatenate([infos[i]["bbox"] for i in range(len(infos))], axis=0)
        infos_pos_enc = np.concatenate([infos[i]["pos_enc"] for i in range(len(infos))], axis=0)
        infos_cls_prob = np.concatenate([infos[i]["cls_prob"] for i in range(len(infos))], axis=0)
        infos_objects = torch.cat([infos[i]["objects"] for i in range(len(infos))], dim=0)
        infos_num_boxes = sum([infos[i]["num_boxes"] for i in range(len(infos))])
        infos_W = infos[-1]["image_width"]
        infos_H = infos[-1]["image_height"]

        new_infos = {'bbox': infos_bbox,
                     'pos_enc': infos_pos_enc,
                     'num_boxes': infos_num_boxes,
                     'objects': infos_objects,
                     'image_width': infos_W,
                     'image_height': infos_H,
                     'cls_prob': infos_cls_prob}
        return [features], [positional_encoding], [new_infos]

    def extract_features(self):
        print("  Extracting features...")
        for i, one_action_data in enumerate(self.data):
            print("    Action indx", i)
            num_frames = len(one_action_data["imgs"])
            k = args.num_key_frames
            step = num_frames//args.num_key_frames
            features, positional_encoding, infos = [], [], []
            print("We have %d frames" % num_frames)
            lb = 0
            hb = 0
            while(hb < num_frames):
                if k != 1:
                    lb = hb
                    hb = min(hb+step, num_frames)
                    print("slice from %d -> %d with %d" % (lb, hb, hb - lb))
                    seq = one_action_data["imgs"][int(lb):int(hb)]
                else:
                    lb = hb
                    hb = num_frames
                    print("slice from %d -> %d with %d" % (lb, hb, hb-lb))
                    seq = one_action_data["imgs"][int(lb):]

                f_extractor = featureExtractor(seq, self.model,
                                               temporal_buffer_size=len(seq))

                feat, pos_enc, info = f_extractor.extract_features()
                feat, pos_enc, info = feat[-1], pos_enc[-1], info[-1]

                while(feat.shape[0] < args.best_features):
                    feat, pos_enc, info = self.add_randomly_selected_box(feat, pos_enc, info)

                features.append(feat)
                positional_encoding.append(pos_enc)
                infos.append(info)

                k -= 1
            # Flatten to one img so that, we have r_0..r_k*nboxes as if it was one image
            # To be applied for the 3 lists above
            features, positional_encoding, infos = self.flatten_to_one_img(features, positional_encoding, infos)
            self.tokenized_data.append({"imgs": {"feat": features, "pos_enco": positional_encoding,
                                                 "spatial": [], "image_mask": [],
                                                 "infos": infos, "co_attention_mask": [],
                                                 "masked_img_labels": [[] for _ in range(len(features))]}})

    def text_tokenizer(self):
        """We add the special tokens to the text (actions)"""
        print("Tokenizing text...")
        for i, one_action_data in enumerate(self.data):
            text = '[CLS]' + one_action_data["desc"][0] + '[SEP]'
            tokenized_text = tokenizer.tokenize(text)
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenized_text)
            self.length_text = len(tokenized_text)
            segment_ids = [0] * len(tokenized_text)
            input_mask = [1] * len(tokenized_text)
            self.max_length = 37
            if len(tokenized_text) < self.max_length:
                # Note here we pad in front of the sentence
                padding = [0] * (self.max_length - len(tokenized_text))
                tokenized_text = tokenized_text + padding
                input_mask += padding
                segment_ids += padding

            self.tokenized_data[i]["desc"] = {"tokenized_text": tokenized_text,
                                              "input_mask": input_mask, "segment_ids": segment_ids,
                                              "modified_token": [], "masked_lm_token": [],
                                              "len_original_text": self.length_text}

    def img_tokenizer(self):
        """Add the spacial token IMG before each set of features extracted from an image"""
        self.extract_features()
        for i, one_action_data in enumerate(self.tokenized_data):
            for j in range(len(one_action_data["imgs"]["feat"])):
                mean_pooled_feat = torch.mean(one_action_data["imgs"]["feat"][j], 0).unsqueeze(
                    0)  # Equivalent to IMG special token
                one_action_data["imgs"]["feat"][j] = torch.cat(
                    (mean_pooled_feat, one_action_data["imgs"]["feat"][j]), dim=0)

                one_action_data["imgs"]["infos"][j]["objects"] = torch.cat(
                    (torch.tensor([-1]), one_action_data["imgs"]["infos"][j]["objects"]), dim=0)
                one_action_data["imgs"]["pos_enco"][j] = torch.cat(
                    (torch.zeros((1, one_action_data["imgs"]["pos_enco"][j].shape[1])), one_action_data["imgs"]["pos_enco"][j]), dim=0)

                boxes = one_action_data["imgs"]["infos"][j]["bbox"]
                image_w = one_action_data["imgs"]["infos"][j]["image_width"]
                image_h = one_action_data["imgs"]["infos"][j]["image_height"]

                image_location = torch.zeros((boxes.shape[0], 5))
                image_location[:, :4] = torch.from_numpy(boxes)
                image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * \
                    (image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))
                image_location[:, 0] = image_location[:, 0] / float(image_w)
                image_location[:, 1] = image_location[:, 1] / float(image_h)
                image_location[:, 2] = image_location[:, 2] / float(image_w)
                image_location[:, 3] = image_location[:, 3] / float(image_h)
                full_image_5D_encoding = torch.FloatTensor([[0, 0, 1, 1, 1]])
                one_action_data["imgs"]["image_mask"].append(torch.tensor(
                    [1] * (int(one_action_data["imgs"]["infos"][j]["num_boxes"]+1))))

                spatial_img_location = torch.cat((full_image_5D_encoding, image_location), dim=0)
                one_action_data["imgs"]["spatial"].append(spatial_img_location)

    def mask_text(self):
        """We will generate 2 new outputs from the Tokenized text: the modified token and the mask_lm_token
            which will be stored in the dictionary in self.tokenized_data["desc"] 
            under the keys (text_id, modified_token, mask_lm_token)"""

        for i, one_action_data in enumerate(self.tokenized_data):

            modified_token = deepcopy(one_action_data["desc"]["tokenized_text"])
            masked_lm_labels = -1*np.ones(len(modified_token)).astype(int)
            type_modification = np.random.choice(np.array(["MASK", "random", "unaltered"]), p=[0.8, 0.1, 0.1])
            length_token = one_action_data["desc"]["len_original_text"] - 2
            num_masked_tokens = int(0.15*length_token)
            top = self.get_top_used_words(0.9)
            if num_masked_tokens < 1:
                num_masked_tokens = 1

            if type_modification == "MASK":
                for i in range(num_masked_tokens):
                    p = np.random.choice(np.array(["non_top", "all"]), p=[0.9, 0.1])
                    if p == "non_top":
                        indx = np.random.randint(length_token)
                        while(modified_token[indx] in top):
                            indx = np.random.randint(length_token)
                    else:
                        indx = np.random.randint(length_token)
                    print(tokenizer.convert_ids_to_tokens(modified_token[indx+1]))
                    modified_token[indx+1] = tokenizer.encode('[MASK]')[0]
                    masked_lm_labels[indx+1] = one_action_data["desc"]["tokenized_text"][indx+1]

            elif type_modification == "random":
                for i in range(num_masked_tokens):
                    p = np.random.choice(np.array(["non_top", "all"]), p=[0.9, 0.1])
                    if p == "non_top":
                        indx = np.random.randint(length_token)
                        while(modified_token[indx] in top):
                            indx = np.random.randint(length_token)
                    else:
                        indx = np.random.randint(length_token)
                    print(tokenizer.convert_ids_to_tokens(modified_token[indx+1]))
                    modified_token[indx+1] = np.random.randint(30522)
                    masked_lm_labels[indx+1] = one_action_data["desc"]["tokenized_text"][indx+1]
            else:
                for i in range(num_masked_tokens):
                    p = np.random.choice(np.array(["non_top", "all"]), p=[0.9, 0.1])
                    if p == "non_top":
                        indx = np.random.randint(length_token)
                        while(modified_token[indx] in top):
                            indx = np.random.randint(length_token)
                    else:
                        indx = np.random.randint(length_token)
                    print(tokenizer.convert_ids_to_tokens(modified_token[indx+1]))
                    masked_lm_labels[indx+1] = one_action_data["desc"]["tokenized_text"][indx+1]

            one_action_data["desc"]["modified_token"] = modified_token
            one_action_data["desc"]["masked_lm_token"] = masked_lm_labels


    def mask_img(self):
        """"We will mask the 15% of the patches features with 90% probability to zeroed features 
        and 10% unalteres"""

        for _, one_action_data in enumerate(self.tokenized_data):
            for f in range(len(one_action_data["imgs"]["feat"])):
                length_feat = one_action_data["imgs"]["feat"][f].shape[0]
                type_modification = np.random.choice(np.array(["zeros", "unaltered"]), p=[0.9, 0.1])
                masked_im_labels = -1*np.ones(length_feat).astype(int)
                num_masked_tokens = int(0.15*(length_feat-1))  # Ignore IMG Token
                if num_masked_tokens < 1:
                    num_masked_tokens = 1
                if type_modification == "zeros":
                    for _ in range(num_masked_tokens):  # Ignore IMG Token
                        i = np.random.randint(length_feat-1)
                        # Masking box j in image f
                        one_action_data["imgs"]["feat"][f][i +
                                                           1] = torch.zeros((1, one_action_data["imgs"]["feat"][f].shape[1]))
                        masked_im_labels[i+1] = one_action_data["imgs"]["infos"][f]["objects"][i+1].item()
                one_action_data["imgs"]["masked_img_labels"][f] = torch.from_numpy(masked_im_labels)

    def build_batch(self, data):
        features_masked, pos_enc, spatial, image_mask, tokenized_text, masked_text, masked_lm_token, input_mask, segment_ids, co_attention_mask, infos, masked_img_labels = [
        ], [], [], [], [], [], [], [], [], [], [], []
        for i in range(len(data)):
            features_masked.append(data[i]["imgs"]["feat"].unsqueeze(0))
            pos_enc.append(data[i]["imgs"]["pos_enco"].unsqueeze(0))
            spatial.append(data[i]["imgs"]["spatial"].unsqueeze(0))
            image_mask.append(data[i]["imgs"]["image_mask"].unsqueeze(0))
            tokenized_text.append(data[i]["desc"]["tokenized_text"].unsqueeze(0))
            masked_text.append(data[i]["desc"]["modified_token"].unsqueeze(0))
            masked_lm_token.append(data[i]["desc"]["masked_lm_token"].unsqueeze(0))
            input_mask.append(data[i]["desc"]["input_mask"].unsqueeze(0))
            segment_ids.append(data[i]["desc"]["segment_ids"].unsqueeze(0))
            co_attention_mask.append(data[i]["imgs"]["co_attention_mask"].unsqueeze(0))
            infos.append(data[i]["imgs"]["infos"]),
            masked_img_labels.append(data[i]["imgs"]["masked_img_labels"].unsqueeze(0))
        features_masked = torch.cat(features_masked, dim=0)
        pos_enc = torch.cat(pos_enc, dim=0)
        spatial = torch.cat(spatial, dim=0)
        image_mask = torch.cat(image_mask, dim=0)
        tokenized_text = torch.cat(tokenized_text, dim=0)
        masked_text = torch.cat(masked_text, dim=0)
        masked_lm_token = torch.cat(masked_lm_token, dim=0)
        input_mask = torch.cat(input_mask, dim=0)
        segment_ids = torch.cat(segment_ids, dim=0)
        co_attention_mask = torch.cat(co_attention_mask, dim=0)
        masked_img_labels = torch.cat(masked_img_labels, dim=0)
        return features_masked, pos_enc, spatial, image_mask, tokenized_text, masked_text, masked_lm_token, input_mask, segment_ids, co_attention_mask, infos, masked_img_labels

    def save_dataloader(self, data):
        torch.save(data, 'DataLoader.pt')

    def load_dataloader(self, path):
        return torch.load(path)

    def get_data_masked_train(self):
        """This function executes tranforms the text into tockens, the extractor of features
        the masking in the text and image"""
        self.img_tokenizer()
        self.text_tokenizer()
        self.mask_text()
        self.mask_img()
        data = self.get_processed_data()
        for instruction, data_point in enumerate(data):
            for type_data_key, type_data_value in data_point.items():  # Here IMGS or DESC
                for key, value in type_data_value.items():
                    if type_data_key == "imgs":
                        if key == "infos":
                            continue
                        if key == "co_attention_mask":
                            type_data_value[key].append(torch.zeros(
                                (data[0]["imgs"]["feat"].shape[0], self.max_length)))
                        if key == "image_mask":
                            cat = value[0]
                            for i in range(1, len(value)):
                                cat = torch.cat((cat, value[i]), dim=0)
                            type_data_value[key] = cat
                        else:
                            type_data_value[key] = torch.cat(value, dim=0)
                    else:
                        type_data_value[key] = torch.tensor(value)
        final_tensor_data = self.build_batch(data)
        if self.save_or_not == True:
            self.save_dataloader(final_tensor_data)

        return final_tensor_data


if __name__ == '__main__':
    frcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    data_loader = DataLoader("json_data.json", "json_token_count.json", frcnn_model, save_or_not=True)

    # to save DataLoader result
    data = data_loader.get_data_masked_train()
