import os
import json
import torch
import cn_clip
import loralib as lora
from cn_clip.clip import load_from_name, available_models
from cn_clip.clip.model import CLIP

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def lora_model():
    r = 16
    alpha = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./model_weight')
    layer_names_dict = model.state_dict().keys()
    module_list = []
    for key in layer_names_dict:
        module_list.append('.'.join(key.split('.')[:-1]))

    #遍历网络结构，如果有"query", "value"对该节点进行替换成lora
    for submodule_key in module_list:
        if submodule_key.split('.')[-1] in ["query", "value"]:
            module_state_dict = model.get_submodule(submodule_key).state_dict()
            #获取该节点的尺寸
            submodule = model.get_submodule(submodule_key)
            lora_layer = lora.Linear(
                submodule.in_features,
                submodule.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=0.1
            )
            #加载预训练权重到lora中
            lora_layer.load_state_dict(module_state_dict,strict=False)
            #lora_layer替代q,v的全连接层
            _set_module(model, submodule_key, lora_layer)
            #设置可训练参数
    return model,preprocess

def set_trainable_params(model):
    for n, p in model.parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

if __name__=="__main__":
    lora_model()