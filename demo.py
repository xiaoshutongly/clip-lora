import numpy as np
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from peft import get_peft_config, PeftModel, get_peft_model, LoraConfig, TaskType

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", cache_dir='../model').to(device)
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", cache_dir='../model')

peft_config = LoraConfig(target_modules=["query", "value"], inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)
model = get_peft_model(model, peft_config)
model.to(device)
model.print_trainable_parameters()

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
def cross_entropy_np(x, y):
    x_softmax = [softmax(x[i]) for i in range(len(x))]
    x_log = [np.log(x_softmax[i][y[i]]) for i in range(len(y))]
    loss = - np.sum(x_log) / len(y)
    return loss

# 测试逻辑
x = [[1.9269, 1.4873, 0.9007, -2.1055]]
y = [[2]]
v1 = cross_entropy_np(x, y)
print(f"v1: {v1}")

def soft_max(z):
    t = np.exp(z)
    n = np.sum(t, axis=1)
    m = np.expand_dims(np.sum(t, axis=1), 1)
    a = np.exp(z) / np.expand_dims(np.sum(t, axis=1), 1)
    return a

Query = np.array([
    [1,0,2],
    [2,2,2],
    [2,1,3]
])

Key = np.array([
    [0,1,1],
    [4,4,0],
    [2,3,1]
])

Value = np.array([
    [1,2,3],
    [2,8,0],
    [2,6,3]
])

scores = Query @ Key.T
print(scores)
scores = soft_max(scores)
print(scores)
out = scores @ Value
print(out)

