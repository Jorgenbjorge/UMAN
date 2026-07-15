from copy import deepcopy
import os
from typing import List, Tuple, Any
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import BertConfig, BertTokenizer
import random
import re
import json
import csv

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class CXRBertTokenizer(BertTokenizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class ALTADataset(Dataset):
    def __init__(self, data_root, is_train, args, max_caption_length: int = 128):
        self.is_train = is_train
        self.max_caption_length = max_caption_length
        
        self.data_root = data_root 
        self.image_dirs = args.image_dirs 
        print(f"[Dataset] Searching for images in subdirectories: {self.image_dirs}")

        self.transform_big = self._build_transform("big")
        self.transform_small = self._build_transform("small")
        
        df = self._read_jsonl() 
        
        all_images = df["image"].tolist()
        all_captions = df["caption"].tolist()
        total_count = len(all_images)
        
        self.images_list = []
        self.captions_list = []
        
        print(f"[Dataset] Total entries in JSONL: {total_count}. Filtering for available images...")
        
        for img_name, caption in zip(all_images, all_captions):
            if self._find_image_path(img_name) is not None:
                self.images_list.append(img_name)
                self.captions_list.append(caption)
        
        found_count = len(self.images_list)
        skipped_count = total_count - found_count
        print(f"[Dataset] Filtering complete. Found {found_count} valid image-caption pairs. Skipped {skipped_count} missing images.")
        
        # 确保 args.bert_path 有效 (已在 main_pretrain.py 中修复)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    def _find_image_path(self, img_name):
        for dir_name in self.image_dirs:
            img_path = os.path.join(self.data_root, dir_name, img_name)
            if os.path.exists(img_path):
                return img_path
        return None

    def _read_jsonl(self):
        if self.is_train:
            jsonl_file = os.path.join(self.data_root, "us_caption_train_qwen3_8b.jsonl")
        else:
            jsonl_file = os.path.join(self.data_root, "us_caption_val_qwen3_8b.jsonl")
        
        print(f"[Dataset] Loading data from: {jsonl_file}")
        
        data = []
        refined_count = 0
        caption_count = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    if 'image' not in item:
                        continue
                    
                    # === 关键修改: 优先使用refined ===
                    caption_text = None
                    if 'refined' in item and item['refined'] and len(item['refined'].strip()) > 0:
                        caption_text = item['refined']
                        refined_count += 1
                    elif 'caption' in item and item['caption'] and len(item['caption'].strip()) > 0:
                        caption_text = item['caption']
                        caption_count += 1
                    else:
                        continue
                    
                    data.append({
                        "image": item["image"], 
                        "caption": caption_text,
                    })
                    
                except json.JSONDecodeError as e:
                    continue

        print(f"[Dataset] Text sources: {refined_count} refined, {caption_count} caption")
        
        if not data:
            raise ValueError(f"No valid data found in {jsonl_file}")
        
        df = pd.DataFrame(data)
        return df

    def _build_transform(self, kind: str):
        """
        针对超声图像的特殊数据增强:
        - 避免过度旋转 (超声探头有固定方向)
        - 增加对比度/亮度增强 (模拟不同设备)
        - 添加高斯模糊 (模拟噪声)
        """
        if kind == "big":
            transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                # === 新增: 超声特定增强 ===
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1)
                ], p=0.5),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.3),
                # 轻微旋转 (不超过10度)
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # 添加随机擦除 (模拟伪影)
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            ])
        elif kind == "small":
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return transform

    def tokenize_caption(self, text: str):
        # 清理文本
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', ' ', text)
        text = re.sub(r' +', ' ', text)

        max_len = self.max_caption_length
        tokenized_output = self.tokenizer(
            text=text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = tokenized_output["input_ids"]
        attention_mask = tokenized_output["attention_mask"]
        
        # === 关键修复: 正确生成MLM标签 ===
        labels = input_ids.clone()
        
        # 随机掩码15%的tokens
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(labels.squeeze().tolist(), already_has_special_tokens=True), 
            dtype=torch.bool
        ).unsqueeze(0)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算被掩码token的loss
        
        # 80%替换为[MASK], 10%随机token, 10%保持原样
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        remaining_indices = masked_indices & ~indices_replaced
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & remaining_indices
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return (
            input_ids,      # 用于MLM的输入 (带mask)
            labels,         # MLM标签 (未mask的是-100)
            tokenized_output["input_ids"],  # 原始IDs (用于CLIP)
            attention_mask,
        )

    def __len__(self):
        return len(self.images_list)

    def get_stats(self):
        available_images = len(self.images_list) 
        caption_lengths = [len(str(caption).split()) for caption in self.captions_list]
        
        stats = {
            "total_samples": len(self),
            "available_images": available_images,
            "is_train": self.is_train,
            "data_root": self.data_root,
            "max_caption_length": self.max_caption_length,
            "avg_caption_length": np.mean(caption_lengths) if caption_lengths else 0,
            "max_actual_caption_length": max(caption_lengths) if caption_lengths else 0,
            "min_caption_length": min(caption_lengths) if caption_lengths else 0
        }
        return stats

    def __getitem__(self, index: int) -> dict:
        """
        === 单视图模式: 只返回1张图像和1个文本 ===
        返回的所有tensor都是 (1, ...) 形状以保持接口一致性
        """
        img_name = self.images_list[index]
        caption = self.captions_list[index]
        
        img_path = self._find_image_path(img_name)
        
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} (index {index}) not found")
        
        img_pil = pil_loader(img_path)
        
        # Tokenize文本
        inputs, labels, ids, attention_mask = self.tokenize_caption(caption)
        
        # 应用图像变换
        img_big = self.transform_big(img_pil)
        img_small = self.transform_small(img_pil)
        
        # === 单视图模式: 添加视图维度但不复制 ===
        # Shape: (1, C, H, W) - 1表示单个视图
        img_stack = img_big.unsqueeze(0)  # (1, C, H, W)
        a_img_stack = img_small.unsqueeze(0)  # (1, C, H, W)
        l_img_stack = img_big.unsqueeze(0)  # (1, C, H, W)
        la_img_stack = img_small.unsqueeze(0)  # (1, C, H, W)
        
        # 文本也添加"视图"维度（实际上是为了接口一致性）
        # Shape: (1, max_len)
        inputs_stack = inputs  # 已经是 (1, max_len)
        labels_stack = labels
        ids_stack = ids
        attention_mask_stack = attention_mask

        return_dict = {
            "img": img_stack,  # (1, C, H, W)
            "a_img": a_img_stack,
            "l_img": l_img_stack,
            "la_img": la_img_stack,
            "inputs": inputs_stack,  # (1, max_len)
            "labels": labels_stack,
            "ids": ids_stack,
            "attention_mask": attention_mask_stack,
        }

        return return_dict
# ==============================================================
# ✅ 轻量版推理数据集 (BUSI_Dataset)
# ==============================================================
class BUSI_Dataset(Dataset):
    """
    用于 BUSI 二分类推理 / 零样本评估。
    从 test.txt 读取文件名，从 labels.csv 读取标签。
    图像位于 all/images/ 下。
    """
    def __init__(self, 
                 image_root="/media/profz/data1/hmd/data/NextGen-UIA/all/images",
                 label_csv="/media/profz/data1/hmd/data/NextGen-UIA/classification/BUSI/labels.csv",
                 split_txt="/media/profz/data1/hmd/data/NextGen-UIA/classification/BUSI/test.txt",
                 transform=None):
        super().__init__()

        self.image_root = image_root
        self.label_csv = label_csv
        self.split_txt = split_txt
        self.transform = transform

        # ===== 1. 读取标签表 =====
        self.labels_dict = {}
        with open(label_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            for row in reader:
                if len(row) < 2:
                    continue
                img_name, label = row[0], row[1]
                self.labels_dict[img_name.strip()] = int(label.strip())

        # ===== 2. 读取测试文件名列表 =====
        with open(split_txt, "r", encoding="utf-8") as f:
            test_ids = [line.strip() for line in f if line.strip()]

        # ===== 3. 匹配图像路径与标签 =====
        self.samples = []
        for img_id in test_ids:
            img_file = os.path.join(image_root, img_id)
            if not os.path.exists(img_file):
                img_file = os.path.join(image_root, f"{img_id}.png")
            if not os.path.exists(img_file):
                img_file = os.path.join(image_root, f"{img_id}.jpg")
            if not os.path.exists(img_file):
                continue

            label = self.labels_dict.get(img_id, None)
            if label is None:
                label = self.labels_dict.get(f"{img_id}.png", None)
            if label is None:
                print(f"[Warning] No label for {img_id}")
                continue

            self.samples.append((img_file, label))

        if len(self.samples) == 0:
            raise ValueError(f"[BUSI_Dataset] No valid samples found in {split_txt}")

        print(f"[BUSI_Dataset] Loaded {len(self.samples)} test samples.")

        # ===== 默认 transform =====
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Failed to read {img_path}: {e}")
            return self.__getitem__((index + 1) % len(self.samples))

        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label