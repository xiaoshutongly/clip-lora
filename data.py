import os
import lmdb
import pickle
import base64
from cn_clip.clip import tokenize,_tokenizer
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from io import BytesIO
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform
def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


def _convert_to_rgb(image):
    return image.convert('RGB')

class LMDBDataset(Dataset):
    def __init__(self,lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        super(LMDBDataset, self).__init__()
        self.lmdb_path = lmdb_path
        self.lmdb_pairs = os.path.join(self.lmdb_path, 'pairs')
        self.lmdb_imgs = os.path.join(self.lmdb_path, "imgs")

        self.env_pairs = lmdb.open(self.lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(self.lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)

        # fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'))
        print("{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples))

        self.dataset_len = self.number_samples
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split
        self.max_txt_length = max_txt_length

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = create_transform(
                input_size=resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment='original',
                interpolation='bicubic',
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __getitem__(self, index):
        sample_index = index % self.number_samples
        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode('utf-8')).tobytes())
        image_id, text_id, raw_text = pair

        image_b64 = self.txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = self.transform(image)

        text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return image, text, eos_index

    def __len__(self):
       return self.dataset_len

def get_dataset(batch_size,data_path,is_train,use_augment,max_txt_length=64,resolution=224):
    dataset = LMDBDataset(
        data_path,
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=use_augment if is_train else False,
        resolution=resolution,
    )
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size
    )
    return dataloader

# data_path = "./datasets/Flickr30k-CN/lmdb/valid/"
# dataloader =get_dataset(batch_size=2,data_path=data_path,is_train=True,use_augment=True,max_txt_length=64,resolution=224)
# for i,(images,texts,eos_index) in enumerate(dataloader):
#     print(1)