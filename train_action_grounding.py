import logging
import json
import torch
from types import SimpleNamespace
from vilbert.vilbert import VILBertActionGrounding, BertConfig
from pytorch_transformers.tokenization_bert import BertTokenizer
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    args = SimpleNamespace(from_pretrained="save/multitask_model/multi_task_model.bin",
                           bert_model="bert-base-uncased",
                           config_file="config/bert_base_6layer_6conect.json",
                           max_seq_length=101,
                           train_batch_size=1,
                           do_lower_case=True,
                           predict_feature=False,
                           seed=42,
                           num_workers=0,
                           baseline=False,
                           img_weight=1,
                           distributed=False,
                           objective=1,
                           visual_target=0,
                           dynamic_attention=False,
                           task_specific_tokens=True,
                           tasks='1',
                           save_name='',
                           in_memory=False,
                           local_rank=-1,
                           split='mteval',
                           clean_train_sets=True,
                           gradient_accumulation_steps=1,
                           num_train_epochs=10.0,
                           train_batch_size=1,
                           start_epoch=0,
                           without_coattention=False
                           )

    config = BertConfig.from_json_file(args.config_file)
    bert_weight_name = json.load(
        open("config/" + args.bert_model + "_weight_name.json", "r")
    )
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    # train_dataset =
    # val_dataset =
    num_train_optimization_steps = int(
        train_dataset.num_dataset
        / args.train_batch_size
        / args.gradient_accumulation_steps
    ) * (args.num_train_epochs - args.start_epoch)

    config.visual_target = args.visual_target

    model = VILBertActionGrounding.from_pretrained(
        args.from_pretrained, config=config, default_gpu=default_gpu
    )
    
    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if key[12:] in bert_weight_name:
                lr = args.learning_rate * 0.1
            else:
                lr = args.learning_rate
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value],
                     "lr": lr,
                     "weight_decay": 0.0}
                ]

            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value],
                     "lr": lr,
                     "weight_decay": 0.01}
                ]
