import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist



def train_one_epoch(model,data,epoch,loss,args, optimizer):
    model.train()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    dataloader = data['train'].dataloader
    for i,(images,texts,eos_index) in range(dataloader):
        #梯度清零
        optimizer.zero_grad()
        # 数据指定设备
        images = images.cuda(args.local_device_rank, non_blocking=True)
        texts = texts.cuda(args.local_device_rank, non_blocking=True)
        image_features, text_features, logit_scale = model(images, texts, args.mask_ratio)
        logit_scale = logit_scale.mean()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.arange(len(logits_per_image)).long()
        ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)

        total_loss = (
                             loss_img(logits_per_image, ground_truth)
                             + loss_txt(logits_per_text, ground_truth)
                     ) / 2

        # acc = None
        # if args.report_training_batch_acc:
        #     i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
        #     t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
        #     acc = {"i2t": i2t_acc, "t2i": t2i_acc}
        total_loss.backward()
        optimizer.step()
