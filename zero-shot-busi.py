# busi_test_only.py
# 只加载已训练权重，在 BUSI test.txt 上做 zero-shot 测试（不训练）
# - split_txt 改为: /media/profz/data1/hmd/data/NextGen-UIA/classification/BUSI/test.txt
# - prompts / 评估逻辑保持与 busi_eval_utils.py 一致
# - 默认不触发可视化（避免 epoch%20==0 时额外画图）

import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer

import model as alta_model
from pretrain_datasets import BUSI_Dataset


# -----------------------------
# Robust checkpoint loader
# -----------------------------
def load_checkpoint_safely(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    兼容你保存的 checkpoint-best_*.pth：
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
        # 只打印少量，避免刷屏
        head = skipped[:20]
        print(f"[Checkpoint] Skipped {len(skipped)} keys due to mismatch/unexpected. e.g.: {head}")
    print(f"[Checkpoint] Missing keys: {len(msg.missing_keys)}")
    if len(msg.missing_keys) > 0:
        print(f"  (show first 20) {msg.missing_keys[:20]}")
    print(f"[Checkpoint] Unexpected keys: {len(msg.unexpected_keys)}")
    if len(msg.unexpected_keys) > 0:
        print(f"  (show first 20) {msg.unexpected_keys[:20]}")

    model.to(device)
    return model


# -----------------------------
# Zero-shot evaluation (BUSI test)
# -----------------------------
@torch.no_grad()
def evaluate_zero_shot_busi_test(model, args, device, epoch=1, save_dir=None, visualize=False):
    """
    BUSI test zero-shot:
    - with alignment: manifold_align.score_pairs(...)
    - w/o alignment: global image-text cosine similarity
    """
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = BUSI_Dataset(
        image_root=args.image_root,
        label_csv=args.label_csv,
        split_txt=args.split_txt,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    prompts = [
        "A high-resolution ultrasound image of a benign breast lesion with smooth margins and regular borders.",
        "A high-resolution ultrasound image of a malignant breast tumor with irregular spiculated margins."
    ]

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    text_inputs_list = []
    for p in prompts:
        ti = tokenizer(
            p, return_tensors="pt",
            padding="max_length", max_length=64, truncation=True
        ).to(device)
        text_inputs_list.append(ti)

    all_preds, all_labels, all_scores = [], [], []

    use_alignment = (
        float(getattr(args, "w_align", 1.0)) > 0.0
        and getattr(args, "ablation_mode", "") != "wo_align"
    )

    if use_alignment:
        print(f"\n[Eval] Running Fine-grained Manifold Alignment Evaluation @ Epoch {epoch}...")
    else:
        print(f"\n[Eval] Running Global Cosine (w/o Alignment) Evaluation @ Epoch {epoch}...")

    for i, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(device)
        B = imgs.shape[0]

        img_global, img_local, _, _, _ = model.image_encoder(imgs, mask_ratio=0.0)
        img_global = F.normalize(img_global, dim=-1)

        batch_scores = []

        for txt_input in text_inputs_list:
            curr_input_ids = txt_input["input_ids"].repeat(B, 1)
            curr_attn_mask = txt_input["attention_mask"].repeat(B, 1)

            txt_embed = model.bert_encoder(
                curr_input_ids,
                curr_attn_mask,
                output_cls_projected_embedding=True,
                return_dict=True
            ).cls_projected_embedding
            txt_global = model.text_proj_adapter(txt_embed)
            txt_global = F.normalize(txt_global, dim=-1)

            if use_alignment:
                txt_output = model.bert_encoder.bert(
                    input_ids=curr_input_ids,
                    attention_mask=curr_attn_mask,
                    return_dict=True
                )
                txt_hidden = txt_output.last_hidden_state
                txt_tokens = model.text_local_proj(txt_hidden[:, 1:, :])
                txt_tokens = F.normalize(txt_tokens, dim=-1)

                score = model.manifold_align.score_pairs(
                    img_global=img_global,
                    img_patches=img_local,
                    txt_global=txt_global,
                    txt_tokens=txt_tokens
                )
            else:
                score = (img_global * txt_global).sum(dim=-1)

            batch_scores.append(score)

        scores_tensor = torch.stack(batch_scores, dim=1)
        probs = torch.softmax(scores_tensor / 0.1, dim=-1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_scores.extend(probs[:, 1].cpu().tolist())

        if i % 10 == 0:
            print(f"   Batch {i}/{len(test_loader)} Processed. Current Acc: {accuracy_score(all_labels, all_preds):.4f}")

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
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        csv_name = "busi_eval_history_local.csv" if use_alignment else "busi_eval_history_wo_align.csv"
        csv_path = os.path.join(save_dir, csv_name)
        file_exists = os.path.exists(csv_path)

        import csv
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"[Eval] History appended to: {csv_path}")

    if use_alignment:
        print(f"\n=== Zero-shot Local Alignment Eval @ Epoch {epoch} ===")
    else:
        print(f"\n=== Zero-shot Global Cosine Eval (w/o Alignment) @ Epoch {epoch} ===")

    print(f"📊 Samples: {len(all_labels)}")
    print(f"✅ Acc: {acc*100:.2f}%  |  📈 AUC: {auc:.4f}")
    print(f"   F1: {f1:.4f}  |  Prec: {prec:.4f}  |  Rec: {rec:.4f}")

    if visualize and use_alignment:
        try:
            from visualization_utils import ALTAVisualizer
            vis_dir = os.path.join(save_dir if save_dir else ".", "attention_maps")
            os.makedirs(vis_dir, exist_ok=True)
            visualizer = ALTAVisualizer(model=model, device=device, save_dir=vis_dir)

            simple_prompts = [
                "A high-resolution ultrasound image of a benign breast lesion with smooth margins.",
                "A high-resolution ultrasound image of a malignant breast tumor with irregular margins."
            ]

            for idx, (imgs, labels) in enumerate(test_loader):
                if idx >= 2:
                    break
                img_sample = imgs[0]
                label = labels[0].item()
                prompt = simple_prompts[label]
                visualizer.visualize_attention_maps(
                    image=img_sample,
                    text_prompt=prompt,
                    save_name=f"epoch{epoch}_sample{idx}_label{label}.png"
                )
        except Exception as e:
            print(f"[Warning] Visualization failed: {e}")

    return metrics


def build_args():
    parser = argparse.ArgumentParser("BUSI test-only zero-shot (ALTA)", add_help=True)

    # ---- paths ----
    parser.add_argument(
        "--checkpoint",
        default="/media/profz/data1/hmd/MTG_newdata/v3/checkpoint-best_combined.pth",
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

    # ---- BUSI dataset ----
    parser.add_argument(
        "--image_root",
        default="/media/profz/data1/hmd/data/NextGen-UIA/all/images",
        type=str
    )
    parser.add_argument(
        "--label_csv",
        default="/media/profz/data1/hmd/data/NextGen-UIA/classification/BUSI/labels.csv",
        type=str
    )
    parser.add_argument(
        "--split_txt",
        default="/media/profz/data1/hmd/data/NextGen-UIA/classification/BUSI/test.txt",  # ✅ 你指定的 test.txt
        type=str
    )

    # ---- runtime ----
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--epoch_tag", default=1, type=int, help="just for logging/csv (avoid 0 to skip auto-vis)")
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
    parser.add_argument("--align_topo_w", default=0.20, type=float)
    parser.add_argument("--align_sparse_w", default=0.02, type=float)
    parser.add_argument("--align_local_floor", default=0.20, type=float)
    parser.add_argument("--align_topk", default=0.30, type=float)

    #parser.add_argument("--w_align", default=0, type=float)
    #parser.add_argument("--ablation_mode", default="wo_align", type=str)

    return parser.parse_args()


def main():
    args = build_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[Device] {device}")

    # build model
    model = alta_model.ALTA_ViT(args=args)
    model = load_checkpoint_safely(model, args.checkpoint, device)

    # eval
    evaluate_zero_shot_busi_test(
        model=model,
        args=args,
        device=device,
        epoch=args.epoch_tag,
        save_dir=args.output_dir,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
