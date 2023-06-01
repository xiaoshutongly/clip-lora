import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist
from data import get_dataset
from torch import optim
from model_lora import lora_model,set_trainable_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def lora_state_dict(model_state_dict):
    return {k: model_state_dict[k] for k in model_state_dict if "lora_" in k}

def save_lora_model(model,checkpoint_path):
    lora_dict = lora_state_dict(model.state_dict())
    torch.save(lora_dict,checkpoint_path)
    print("Save lora model to " + checkpoint_path)
    print("Lora dict is " + str(lora_dict.keys()))

def optimizer(model):
    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.001},
        ],
        lr=5e-5,
    )
    return optimizer

def train_one_epoch(model, data_path, optimizer,batch_size,checkpoint_path):
    model.train()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.to(device)
    loss_txt = loss_txt.to(device)

    #data_path = "./datasets/Flickr30k-CN/lmdb/valid/"
    dataloader = get_dataset(batch_size=batch_size, data_path=data_path, is_train=True, use_augment=True, max_txt_length=64,
                             resolution=224)
    for i,(images,texts,eos_index) in enumerate(dataloader):
        #梯度清零
        optimizer.zero_grad()
        #数据指定设备
        images = images.to(device)#.to(torch.half)
        texts = texts.to(device)
        image_features, text_features, logit_scale = model(images, texts)
        logit_scale = logit_scale.mean()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.arange(len(logits_per_image)).long()
        ground_truth = ground_truth.cuda()

        total_loss = (
                             loss_img(logits_per_image, ground_truth)
                             + loss_txt(logits_per_text, ground_truth)
                     ) / 2

        total_loss.backward()
        optimizer.step()
        print(i)
    save_lora_model(model, checkpoint_path)
if __name__== "__main__":
    model = lora_model().to(device)
    set_trainable_params(model)
    convert_models_to_fp32(model)
    optimizer = optimizer(model)
    data_path = "./datasets/Flickr30k-CN/lmdb/valid/"
    train_one_epoch(model, data_path, optimizer,batch_size=64,checkpoint_path='lora_weights.pt')