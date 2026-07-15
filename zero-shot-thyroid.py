# thyroid_test_only.py
# 只加载已训练权重，在 甲状腺良/恶性（二分类）测试集（按文件夹 0/1 分好类，bmp）上做 zero-shot 测试（不训练）
# - 数据集目录：
#   /media/profz/data1/hmd/train/train/train/train/Classification/Two/thyroid/cls/test/
#     ├── 0   (benign)
#     └── 1   (malignant)
# - 评估逻辑保持与你 BUSI/HCC 脚本一致：
#   manifold_align.score_pairs + softmax(scores/0.1)
# - 提示词：提供多个 prompt set，可用 --prompt_set 1/2/3/4 切换

import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer
from PIL import Image

import model as alta_model


# -----------------------------
# Dataset: folder-based binary classification (0/1)
# -----------------------------
class ThyroidFolderDataset(Dataset):
    """
    Folder structure:
      root/
        0/  -> label 0 (benign)
        1/  -> label 1 (malignant)
    Images are .bmp (but we also allow common extensions for robustness).
    """
    IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")

    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset root not found: {root_dir}")

        self.class_to_idx = {"0": 0, "1": 1}

        self.samples = []
        for cls_name, label in self.class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")

            for fn in sorted(os.listdir(cls_dir)):
                fp = os.path.join(cls_dir, fn)
                if os.path.isfile(fp) and fn.lower().endswith(self.IMG_EXTS):
                    self.samples.append((fp, label))

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found under: {root_dir}")

        n0 = sum(1 for _, y in self.samples if y == 0)
        n1 = sum(1 for _, y in self.samples if y == 1)
        print(f"[ThyroidFolderDataset] Loaded {len(self.samples)} samples. benign(0)={n0}, malignant(1)={n1}")

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


# -----------------------------
# Robust checkpoint loader
# -----------------------------
def load_checkpoint_safely(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    兼容 checkpoint-best_*.pth：
    - 支持 ckpt['model'] / ckpt['state_dict'] / 直接是 state_dict
    - 自动跳过 shape 不匹配的权重（常见：BERT vocab / MLM head）
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Checkpoint] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt if isinstance(ckpt, dict) else {}

    model_state = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)

    msg = model.load_state_dict(filtered, strict=False)
    print("[Checkpoint] Load finished.")
    if len(skipped) > 0:
        print(f"[Checkpoint] Skipped {len(skipped)} keys due to mismatch/unexpected. e.g.: {skipped[:20]}")
    print(f"[Checkpoint] Missing keys: {len(msg.missing_keys)}")
    if len(msg.missing_keys) > 0:
        print(f"  (show first 20) {msg.missing_keys[:20]}")
    print(f"[Checkpoint] Unexpected keys: {len(msg.unexpected_keys)}")
    if len(msg.unexpected_keys) > 0:
        print(f"  (show first 20) {msg.unexpected_keys[:20]}")

    model.to(device)
    return model


# -----------------------------
# Prompt sets (benign=0, malignant=1)
# -----------------------------
def get_prompt_sets():
    # benign = 0, malignant = 1
    prompt_sets = {
        # -----------------------------
        # 1) 最推荐：短句 + 核心征象
        # -----------------------------
        1: [
"A thyroid ultrasound image of a benign thyroid lesion such as a colloid nodule or adenoma, with smooth well-circumscribed borders and no suspicious calcifications.",
"A thyroid ultrasound image of thyroid carcinoma, with irregular borders, microcalcifications, and a taller-than-wide appearance."],

        # -----------------------------
        # 2) TI-RADS 风格：更贴近甲状腺超声判读
        # -----------------------------
        2: [
            "A thyroid ultrasound image showing a probably benign nodule with smooth well-defined margins, oval shape, and relatively homogeneous echotexture.",
            "A thyroid ultrasound image showing a suspicious malignant nodule with irregular or ill-defined margins, taller-than-wide shape, marked hypoechogenicity, and punctate echogenic foci."
        ],

        # -----------------------------
        # 3) 病理指向型：良性结节 vs 甲状腺癌
        # -----------------------------
        3: [
            "A thyroid ultrasound image of a benign thyroid lesion such as a colloid nodule or adenoma, with smooth borders and no suspicious calcifications.",
            "A thyroid ultrasound image of thyroid carcinoma, with irregular borders, microcalcifications, and a taller-than-wide appearance."
        ],

        # -----------------------------
        # 4) 回声与边界联合型：不过度堆特征
        # -----------------------------
        4: [
            "A thyroid ultrasound image of a benign nodule with regular margins, wider-than-tall shape, and relatively uniform internal echoes.",
            "A thyroid ultrasound image of a malignant nodule with irregular margins, non-parallel taller-than-wide shape, and heterogeneous hypoechoic internal echoes."
        ],

        # -----------------------------
        # 5) 更自然的医学描述句
        # -----------------------------
        5: [
            "Thyroid ultrasound showing a benign nodule: oval, well-circumscribed, wider-than-tall, and without suspicious punctate echogenic foci.",
            "Thyroid ultrasound showing a malignant nodule: irregular, taller-than-wide, hypoechoic, and containing suspicious punctate echogenic foci."
        ],

        # -----------------------------
        # 6) 极简二分类型：适合先看 prompt 是否过长导致退化
        # -----------------------------
        6: [
            "Benign thyroid nodule on ultrasound.",
            "Malignant thyroid nodule on ultrasound."
        ],

        # -----------------------------
        # 7) 中等长度、通常比较稳
        # -----------------------------
        7: [
            "An ultrasound image of a benign thyroid nodule with smooth circumscribed margins and a parallel wider-than-tall shape.",
            "An ultrasound image of a malignant thyroid nodule with irregular margins, non-parallel taller-than-wide shape, and punctate echogenic foci."
        ],

        # -----------------------------
        # 8) 如果你的数据里乳头状癌较多，这组可能有效
        # -----------------------------
        8: [
            "A thyroid ultrasound image of a benign thyroid nodule with smooth margins and no suspicious malignant features.",
            "A thyroid ultrasound image of papillary thyroid carcinoma with irregular margins, microcalcifications, and a taller-than-wide shape."
        ],
    }
    return prompt_sets


# -----------------------------
# Zero-shot evaluation (thyroid)
# -----------------------------
@torch.no_grad()
def evaluate_zero_shot_thyroid_test(model, args, device, epoch=1, save_dir=None, visualize=False):
    """
    与 BUSI/HCC 脚本一致的评估逻辑：
    - 图像：image_encoder(mask_ratio=0.0) 得到 global + patches
    - 文本：BERT CLS + token 投影
    - 分数：manifold_align.score_pairs(img_global, img_patches, txt_global, txt_tokens)
    - probs：softmax(scores / 0.1)
    - 指标：acc/prec/rec/f1/auc
    - 保存：busi_eval_history_local.csv（保持不变）
    """
    model.eval()

    # 1) 数据加载（folder-based 0/1）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = ThyroidFolderDataset(root_dir=args.test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 2) Prompts
    prompt_sets = get_prompt_sets()
    if args.prompt_set not in prompt_sets:
        raise ValueError(f"--prompt_set must be one of {sorted(prompt_sets.keys())}, got {args.prompt_set}")
    PROMPTS = prompt_sets[args.prompt_set]

    print(f"[Prompts] Using prompt_set={args.prompt_set}")
    print(f"  (0) {PROMPTS[0]}")
    print(f"  (1) {PROMPTS[1]}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    text_inputs_list = []
    for p in PROMPTS:
        ti = tokenizer(
            p, return_tensors="pt",
            padding="max_length", max_length=64, truncation=True
        ).to(device)
        text_inputs_list.append(ti)

    all_preds, all_labels, all_scores = [], [], []

    print(f"\n[Eval] Running Fine-grained Manifold Alignment Evaluation @ Epoch {epoch}...")
    for i, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        B = imgs.shape[0]

        # A) 图像全局+局部
        img_global, img_local, _, _, _ = model.image_encoder(imgs, mask_ratio=0.0)

        batch_scores = []

        # B) 对每个 prompt 计算匹配分
        for txt_input in text_inputs_list:
            curr_input_ids = txt_input["input_ids"].repeat(B, 1)
            curr_attn_mask = txt_input["attention_mask"].repeat(B, 1)

            # 文本全局
            txt_embed = model.bert_encoder(
                curr_input_ids,
                curr_attn_mask,
                output_cls_projected_embedding=True,
                return_dict=True
            ).cls_projected_embedding
            txt_global = model.text_proj_adapter(txt_embed)
            txt_global = F.normalize(txt_global, dim=-1)

            # 文本局部 tokens
            txt_output = model.bert_encoder.bert(
                input_ids=curr_input_ids,
                attention_mask=curr_attn_mask,
                return_dict=True
            )
            txt_hidden = txt_output.last_hidden_state  # [B, L, 768]
            txt_tokens = model.text_local_proj(txt_hidden[:, 1:, :])  # [B, L-1, D]
            txt_tokens = F.normalize(txt_tokens, dim=-1)

            # 流形对齐得分
            score = model.manifold_align.score_pairs(
                img_global=img_global,
                img_patches=img_local,
                txt_global=txt_global,
                txt_tokens=txt_tokens
            )  # [B]
            batch_scores.append(score)

        # C) 预测（保持 softmax(scores/0.1) 不变）
        scores_tensor = torch.stack(batch_scores, dim=1)  # [B, 2]
        probs = torch.softmax(scores_tensor / 0.1, dim=-1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        all_scores.extend(probs[:, 1].detach().cpu().tolist())  # class-1(malignant) prob for AUC

        if i % 10 == 0:
            print(f"   Batch {i}/{len(test_loader)} Processed. Current Acc: {accuracy_score(all_labels, all_preds):.4f}")

    # 4) 指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception as e:
        print(f"[Warning] AUC calculation failed: {e}")
        auc = 0.5

    metrics = {
        "epoch": epoch,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "prompt_set": args.prompt_set
    }

    # 5) 保存记录（保持与之前一致的文件名；加一列 prompt_set）
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "busi_eval_history_local.csv")
        file_exists = os.path.exists(csv_path)

        import csv
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"[Eval] History appended to: {csv_path}")

    print(f"\n=== Zero-shot Local Alignment Eval @ Epoch {epoch} ===")
    print(f"Samples: {len(all_labels)} | prompt_set={args.prompt_set}")
    print(f"Acc: {acc*100:.2f}%  |  AUC: {auc:.4f}")
    print(f"F1: {f1:.4f}  |  Prec: {prec:.4f}  |  Rec: {rec:.4f}")

    # 默认不做可视化（需要再打开 --visualize）
    if visualize:
        try:
            from visualization_utils import ALTAVisualizer
            vis_dir = os.path.join(save_dir if save_dir else ".", "attention_maps")
            os.makedirs(vis_dir, exist_ok=True)
            visualizer = ALTAVisualizer(model=model, device=device, save_dir=vis_dir)

            simple_prompts = PROMPTS  # 直接用当前 prompt set

            for idx, (imgs, labels) in enumerate(test_loader):
                if idx >= 2:
                    break
                img_sample = imgs[0]
                label = labels[0].item()
                prompt = simple_prompts[label]
                visualizer.visualize_attention_maps(
                    image=img_sample,
                    text_prompt=prompt,
                    save_name=f"thyroid_epoch{epoch}_set{args.prompt_set}_sample{idx}_label{label}.png"
                )
        except Exception as e:
            print(f"[Warning] Visualization failed: {e}")

    return metrics


def build_args():
    parser = argparse.ArgumentParser("Thyroid benign/malignant test-only zero-shot (ALTA)", add_help=True)

    # ---- paths ----
    parser.add_argument(
        "--checkpoint",
        default="/media/profz/data1/hmd/MTG_newdata/v1/checkpoint-8.pth",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        default="/media/profz/data1/hmd/MTG_new/v2_2/output_dir",
        type=str
    )
    parser.add_argument( 
        "--bert_path",
        default="/media/profz/data1/hmd/Bio_ClinicalBERT",
        type=str
    )
    parser.add_argument(
        "--mae_path",
        default="/media/profz/data1/hmd/ALTA/vision_encoder_weights/MRM.pth",
        type=str
    )

    # ---- Thyroid folder dataset ----
    parser.add_argument(
        "--test_dir",
        default="/media/profz/data1/hmd/train/train/train/train/Classification/Two/thyroid/cls/test",
        type=str
    )

    parser.add_argument(
        "--prompt_set",
        default=3,
        type=int,
        help="choose prompt set: 1/2/3/4/5/6/7/8"
    )

    # ---- runtime ----
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--epoch_tag", default=1, type=int, help="just for logging/csv")
    parser.add_argument("--visualize", action="store_true", help="enable attention map visualization")

    # ---- model hparams (match your training defaults) ----
    parser.add_argument("--proj_dim", default=512, type=int)
    parser.add_argument("--adapter_type", default="normal", type=str)
    parser.add_argument("--adapter_dim", default=256, type=int)
    parser.add_argument("--adapter_rate", default=0.5, type=float)
    parser.add_argument("--adapter_mlp_ratio", default=0.25, type=float)
    parser.add_argument("--adapter_t_range", default=5, type=float)

    # ---- manifold alignment hparams (used in model.py) ----
    parser.add_argument("--align_temp", default=0.07, type=float)
    parser.add_argument("--align_topo_w", default=0.15, type=float)
    parser.add_argument("--align_sparse_w", default=0.02, type=float)
    parser.add_argument("--align_local_floor", default=0.20, type=float)
    parser.add_argument("--align_topk", default=0.30, type=float)

    return parser.parse_args()


def main():
    args = build_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if (torch.cuda.is_available() and args.device != "cpu") else "cpu")
    print(f"[Device] {device}")

    model = alta_model.ALTA_ViT(args=args)
    model = load_checkpoint_safely(model, args.checkpoint, device)

    evaluate_zero_shot_thyroid_test(
        model=model,
        args=args,
        device=device,
        epoch=args.epoch_tag,
        save_dir=args.output_dir,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
