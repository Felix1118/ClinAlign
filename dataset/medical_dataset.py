import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from .utils import my_pre_caption
import os


CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
    'Aorta',
    'Bone/Spine',
    'Hemidiaphragm',
    'Lung Volume'
]


STATE_PROMPTS = {
    0: None,  # 跳过
    1: "is present in the image.",
    2: "is absent in the image.",
    3: "is uncertain in the image."
}

conditions_dict = {i: condition for i, condition in enumerate(CONDITIONS)}

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]


class generation_train(Dataset):
    def __init__(self,detect_transform, transform, image_root, ann_root, tokenizer, max_words=100, dataset='mimic_cxr', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation['train']
        self.transform = transform
        self.detect_transform=detect_transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args
        with open('/root/autodl-tmp/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)

        #iu_xray Dataset
        #image_path = ann['image_path']
        #image_frontal= Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        #image_lateral = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
        #image_frontal = self.transform(image_frontal)
        #image_lateral = self.transform(image_lateral)
        #image = torch.stack((image_frontal, image_lateral), 0)
        
        cls_labels = ann['labels']
        prefix_prompt = [SCORES[l] for l in cls_labels]
        prefix_prompt = ' '.join(prefix_prompt)+' '
        prompts = []
        for index, label in enumerate(cls_labels):
             if label == 0:
                continue
             disease_name = conditions_dict[index]
             prompt_template = STATE_PROMPTS.get(label, "status unknown in the image.")
             prompt = f"{disease_name} {prompt_template}"
             prompts.append(prompt)
        
        final_prompt=' '.join(prompts)+' '
        caption= prefix_prompt+my_pre_caption(ann['report'], self.max_words)
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        clip_indices = ann['clip_indices'][:self.args.clip_k]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()
        return image, final_prompt ,caption, cls_labels, clip_memory
    
class generation_eval(Dataset):
    def __init__(self,detect_transform, transform, image_root, ann_root, tokenizer, max_words=100, split='val', dataset='mimic_cxr', args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.detect_transform=detect_transform
        if dataset == 'mimic_cxr':
            self.ann = self.annotation[split]
        else: # IU
            self.ann = self.annotation
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
        with open('/root/autodl-tmp/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)
        # iu_xray Dataset
        # image_path = ann['image_path']
        # image_frontal= Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        # image_lateral = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
        # image_frontal = self.transform(image_frontal)
        # image_lateral = self.transform(image_lateral)
        # image = torch.stack((image_frontal, image_lateral), 0)
        caption = my_pre_caption(ann['report'], self.max_words)
        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels))
        clip_indices = ann['clip_indices'][:self.args.clip_k]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()
        return image, caption, cls_labels, clip_memory
    


## iu_xray
# import json
# import os
# import torch
# import numpy as np

# from torch.utils.data import Dataset

# from PIL import Image
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None

# from .utils import my_pre_caption,pre_caption
# import os
# import cv2

# CONDITIONS = [
#     'enlarged cardiomediastinum',
#     'cardiomegaly',
#     'lung opacity',
#     'lung lesion',
#     'edema',
#     'consolidation',
#     'pneumonia',
#     'atelectasis',
#     'pneumothorax',
#     'pleural effusion',
#     'pleural other',
#     'fracture',
#     'support devices',
#     'no finding',
#     'Aorta',
#     'Bone/Spine',
#     'Hemidiaphragm',
#     'Lung Volume'
# ]


# # 预先定义状态映射
# STATE_PROMPTS = {
#     0: None,  # 跳过
#     1: "is present in the image.",
#     2: "is absent in the image.",
#     3: "is uncertain in the image."
# }

# conditions_dict = {i: condition for i, condition in enumerate(CONDITIONS)}

# SCORES = [
# '[BLA]',
# '[POS]',
# '[NEG]',
# '[UNC]'
# ]


# class generation_train(Dataset):
#     def __init__(self, transform, image_root, ann_root, tokenizer, max_words=100, dataset='mimic_cxr', args=None):

#         self.annotation = json.load(open(os.path.join(ann_root),'r'))
#         self.ann = self.annotation['train']
#         self.transform = transform
#         self.image_root = image_root
#         self.tokenizer = tokenizer
#         self.max_words = max_words
#         self.dataset = dataset
#         self.args = args

#     def __len__(self):
#         return len(self.ann)

#     def __getitem__(self, index):

#         ann = self.ann[index]
#         image_path = ann['image_path']
#         image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
#         image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
#         image_1 = self.transform(image_1)
#         image_2 = self.transform(image_2)
#         image = torch.stack((image_1, image_2), 0)
#         cls_labels = ann['label']

#         prefix_prompt = [SCORES[l] for l in cls_labels]
#         prefix_prompt = ' '.join(prefix_prompt)+' '
#         # prefix_prompt='a picture of'
#         prompts = []
#         for index, label in enumerate(cls_labels):
#              if label == 0:
#                 continue

#              disease_name = conditions_dict[index]
#              prompt_template = STATE_PROMPTS.get(label, "status unknown in the image.")
#              prompt = f"{disease_name} {prompt_template}"
#              prompts.append(prompt)

#         final_prompt=' '.join(prompts)+' '
#         # print(final_prompt)

#         caption=prefix_prompt +my_pre_caption(ann['report'], 90)
#         # print(caption)
#         cls_labels = torch.from_numpy(np.array(cls_labels)).long()
#         return image, final_prompt ,caption, cls_labels

#         # return transformed_image,image,prompt, caption, cls_labels, clip_memory

# class generation_eval(Dataset):
#     def __init__(self, transform, image_root, ann_root, tokenizer, max_words=100, split='val', dataset='mimic_cxr', args=None):
#         self.annotation = json.load(open(os.path.join(ann_root), 'r'))
#         self.ann = self.annotation[split]
#         self.transform = transform
#         self.max_words = max_words
#         self.image_root = image_root
#         self.tokenizer = tokenizer
#         self.dataset = dataset
#         self.args = args

#     def __len__(self):
#         return len(self.ann)

#     def __getitem__(self, index):

#         ann = self.ann[index]
#         image_path = ann['image_path']
#         image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
#         image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
#         image_1 = self.transform(image_1)
#         image_2 = self.transform(image_2)
#         image = torch.stack((image_1, image_2), 0)
#         caption = my_pre_caption(ann['report'], 90)
#         cls_labels = ann['label']
#         cls_labels = torch.from_numpy(np.array(cls_labels))

#         return image, caption, cls_labels