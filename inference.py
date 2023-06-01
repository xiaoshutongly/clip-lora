import torch
from PIL import Image
from model_lora import lora_model
import cn_clip.clip as clip
from train import convert_models_to_fp32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
model,preprocess  = lora_model()
model = model.to(device)
convert_models_to_fp32(model)
model_state_dict = model.state_dict()

lora_dict = torch.load('lora_weights.pt')
model_state_dict.update(lora_dict)
model.load_state_dict(model_state_dict)

model.eval()
image = preprocess(Image.open("pokemon.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
