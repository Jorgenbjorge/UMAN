import os
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer

import model as alta_model


class LymphNodeFoldDataset(Dataset):
    """
    使用 5 折 CSV 划分的数据集。
    - fold_number == val_fold -> 验证集
    - fold_number != val_fold -> 训练集（这里只保留接口，当前脚本实际跑 val）
    """
    def __init__(self, image_root, csv_path, val_fold=0, split="val", transform=None):
        self.image_root = image_root
        self.csv_path = csv_path
        self.val_fold = val_fold
        self.split = split
        self.transform = transform

        df = pd.read_csv(csv_path)
        required_cols = {"file_name", "fold_number", "category"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(
                f"CSV must contain columns {required_cols}, but got {list(df.columns)}"
            )

        if split == "val":
            df = df[df["fold_number"] == val_fold].reset_index(drop=True)
        elif split == "train":
            df = df[df["fold_number"] != val_fold].reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.samples = []
        for _, row in df.iterrows():
            file_name = str(row["file_name"])
            label = int(row["category"])
            img_path = os.path.join(image_root, file_name)
            self.samples.append((img_path, label, file_name))

        print(f"[LymphNodeFoldDataset] Loaded {len(self.samples)} {split} samples from fold {val_fold}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _clean_state_dict_keys(state_dict):
    """兼容 DDP / 不同保存方式：去掉常见前缀 module. / model."""
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_sd[nk] = v
    return new_sd




def _extract_feature_tensor(x, name="feature"):
    """兼容不同工程接口：张量 / tuple / list / dict / ModelOutput。"""
    if torch.is_tensor(x):
        return x

    if isinstance(x, (tuple, list)):
        # 优先返回第一个张量项；若有 2D tensor，优先取 2D 特征
        tensor_items = [t for t in x if torch.is_tensor(t)]
        if not tensor_items:
            raise TypeError(f"{name} returned {type(x)}, but no tensor item was found.")
        for t in tensor_items:
            if t.ndim == 2:
                return t
        return tensor_items[0]

    if isinstance(x, dict):
        preferred_keys = [
            "cls_projected_embedding", "projected_embedding",
            "cls_embedding", "embedding", "features", "feature"
        ]
        for k in preferred_keys:
            if k in x and torch.is_tensor(x[k]):
                return x[k]
        for v in x.values():
            if torch.is_tensor(v):
                return v
        raise TypeError(f"{name} returned dict, but no tensor value was found.")

    # 兼容 transformers / 自定义输出对象
    preferred_attrs = [
        "cls_projected_embedding", "projected_embedding",
        "cls_embedding", "embedding", "features", "feature"
    ]
    for attr in preferred_attrs:
        if hasattr(x, attr):
            v = getattr(x, attr)
            if torch.is_tensor(v):
                return v

    raise TypeError(f"Unsupported {name} type: {type(x)}")

def load_checkpoint_to_model(model, ckpt_path, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = _clean_state_dict_keys(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print("[Checkpoint] Loaded.")
    print(f"[Checkpoint] Missing keys: {len(msg.missing_keys)}")
    print(f"[Checkpoint] Unexpected keys: {len(msg.unexpected_keys)}")
    return model


@torch.no_grad()
def evaluate_zero_shot_lymph_node_fold(model, args, device):
    """
    参考 zero-shot-busi 的整体流程：
    仅把数据集替换为淋巴结 5 折划分，并使用 fold 0 作为验证集。
    """
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = LymphNodeFoldDataset(
        image_root=args.image_root,
        csv_path=args.csv_path,
        val_fold=args.val_fold,
        split="val",
        transform=transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 0: metastatic lymph nodes, 1: lymphoma, 2: benign lymph nodes
    prompts = [
        "Malignant metastatic lymph node on ultrasound: loss of normal hilum, cortical thickening or bulging, round morphology, heterogeneous internal echoes, short axis enlargement.",
        "Lymphoma lymph node on ultrasound: uniformly hypoechoic with pseudo-cystic or reticulated texture, markedly enlarged, homogeneous echo pattern, complete hilum absence.",
        "Normal benign lymph node on ultrasound: preserved hyperechoic fatty hilum, thin symmetric cortex, oval kidney-bean shape, homogeneous cortical echoes, normal hilar blood flow.",
    ]

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None or max_len > 100000:
        txt_inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(device)
    else:
        txt_inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)

    txt_embed = model.bert_encoder(
        txt_inputs["input_ids"],
        txt_inputs["attention_mask"],
        output_cls_projected_embedding=True,
        return_dict=True,
    ).cls_projected_embedding
    txt_embed = model.text_proj_adapter(txt_embed)
    txt_embed = F.normalize(txt_embed, dim=-1)  # [3, D]

    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in val_loader:
        imgs = imgs.to(device)

        img_out = model.forward_image_feature(imgs, mask_ratio=0.0)
        img_feats = _extract_feature_tensor(img_out, name="forward_image_feature")
        img_feats = F.normalize(img_feats, dim=-1)

        logits = img_feats @ txt_embed.T  # [B, 3]
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    metrics = {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "auc_macro_ovr": auc,
        "num_samples": len(all_labels),
    }
    return metrics



def get_args():
    parser = argparse.ArgumentParser("Lymph-node zero-shot evaluation on fold-0")

    parser.add_argument(
        "--ckpt",
        default="/media/profz/data1/hmd/MTG_new/v2_2/output_dir/checkpoint-best_combined.pth",
        type=str,
    )

    parser.add_argument(
        "--csv_path",
        default="/media/profz/data1/hmd/2d/cls/5fold_cross_validation.csv",
        type=str,
    )
    parser.add_argument(
        "--image_root",
        default="/media/profz/data1/hmd/2d/cls/train/image",
        type=str,
    )
    parser.add_argument("--val_fold", default=0, type=int)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--proj_dim", default=512, type=int)
    parser.add_argument("--mae_path", default="/media/profz/data1/hmd/ALTA/vision_encoder_weights/MRM.pth", type=str)
    parser.add_argument("--bert_path", default="/media/profz/data1/hmd/Bio_ClinicalBERT", type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    return parser.parse_args()



def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = alta_model.ALTA_ViT(args=args).to(device)
    model = load_checkpoint_to_model(model, args.ckpt, device=device)

    metrics = evaluate_zero_shot_lymph_node_fold(model, args, device)

    print("\n================ Lymph Node Zero-shot Eval ================")
    print(f"CSV     : {args.csv_path}")
    print(f"ImageDir: {args.image_root}")
    print(f"Val Fold: {args.val_fold}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>16s}: {v:.6f}")
        else:
            print(f"{k:>16s}: {v}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
