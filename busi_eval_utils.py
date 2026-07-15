import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pretrain_datasets import BUSI_Dataset
from transformers import BertTokenizer
import csv
import os
import numpy as np

def evaluate_zero_shot_busi(model, args, device, epoch, save_dir):
    """
    Zero-shot 评估：
    - 默认：使用流形对齐后的局部相似度评分
    - w/o alignment：退化为纯全局 cosine 匹配
    """
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = BUSI_Dataset(
        image_root="/media/profz/data1/hmd/data/NextGen-UIA/all/images",
        label_csv="/media/profz/data1/hmd/data/NextGen-UIA/classification/BUSI/labels.csv",
        split_txt="/media/profz/data1/hmd/data/NextGen-UIA/classification/BUSI/test.txt",
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    prompts = [
        "A high-resolution ultrasound image of a benign breast lesion with smooth margins and regular borders.",
        "A high-resolution ultrasound image of a malignant breast tumor with irregular spiculated margins."
    ]

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    text_inputs_list = []
    for p in prompts:
        ti = tokenizer(
            p,
            return_tensors="pt",
            padding='max_length',
            max_length=64,
            truncation=True
        ).to(device)
        text_inputs_list.append(ti)

    all_preds, all_labels, all_scores = [], [], []

    use_alignment = (
        float(getattr(args, 'w_align', 1.0)) > 0.0
        and getattr(args, 'ablation_mode', '') != 'wo_align'
    )

    if use_alignment:
        print(f"\n[Eval] Running Fine-grained Manifold Alignment Evaluation @ Epoch {epoch}...")
    else:
        print(f"\n[Eval] Running Global Cosine (w/o Alignment) Evaluation @ Epoch {epoch}...")

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # 图像特征
            img_global, img_local, _, _, _ = model.image_encoder(imgs, mask_ratio=0.0)
            img_global = F.normalize(img_global, dim=-1)

            batch_scores = []

            for txt_input in text_inputs_list:
                curr_input_ids = txt_input['input_ids'].repeat(B, 1)
                curr_attn_mask = txt_input['attention_mask'].repeat(B, 1)

                # 文本全局特征
                txt_embed = model.bert_encoder(
                    curr_input_ids,
                    curr_attn_mask,
                    output_cls_projected_embedding=True,
                    return_dict=True
                ).cls_projected_embedding
                txt_global = model.text_proj_adapter(txt_embed)
                txt_global = F.normalize(txt_global, dim=-1)

                if use_alignment:
                    # 文本局部特征仅在 alignment 模式下需要
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
        'epoch': epoch,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }

    csv_name = "busi_eval_history_local.csv" if use_alignment else "busi_eval_history_wo_align.csv"
    csv_path = os.path.join(save_dir, csv_name)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    if use_alignment:
        print(f"\n=== Zero-shot Local Alignment Eval @ Epoch {epoch} ===")
    else:
        print(f"\n=== Zero-shot Global Cosine Eval (w/o Alignment) @ Epoch {epoch} ===")

    print(f"📊 Samples: {len(all_labels)}")
    print(f"✅ Acc: {acc*100:.2f}%  |  📈 AUC: {auc:.4f}")
    print(f"   F1: {f1:.4f}  |  Prec: {prec:.4f}  |  Rec: {rec:.4f}")

    if use_alignment and epoch % 20 == 0:
        try:
            from visualization_utils import ALTAVisualizer

            visualizer = ALTAVisualizer(
                model=model,
                device=device,
                save_dir=os.path.join(save_dir, 'attention_maps')
            )

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

                try:
                    visualizer.visualize_attention_maps(
                        image=img_sample,
                        text_prompt=prompt,
                        save_name=f'epoch{epoch}_sample{idx}_label{label}.png'
                    )
                except Exception as e:
                    print(f"[Warning] Attention visualization failed: {e}")
                    break
        except ImportError:
            print(f"[Warning] visualization_utils module not found, skipping visualization")
        except Exception as e:
            print(f"[Warning] Visualization setup failed: {e}")

    model.train()
    return metrics