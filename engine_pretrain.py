import math
import sys
from typing import Iterable
import torch
import torch.nn.functional as F
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

from visualization_utils import ALTAVisualizer


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    mask_ratio = args.mask_ratio

    # 创建可视化保存目录（保持你的目录名不变）
    if misc.is_main_process():
        vis_dir = os.path.join(args.output_dir, 'alignment_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    else:
        vis_dir = None

    # ✅ 只在主进程、且需要时初始化一次 visualizer（避免每1000步重复构造）
    visualizer = None
    if misc.is_main_process() and vis_dir is not None:
        try:
            visualizer = ALTAVisualizer(model=model, device=device, save_dir=vis_dir)
        except Exception as e:
            print(f"[WARNING] Failed to init ALTAVisualizer: {e}")
            visualizer = None

    # 🔥 用于统计每个epoch的平均辅助损失
    epoch_similarities = []
    epoch_weights = []

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # 加载数据
        img_stack = batch['img'].to(device, non_blocking=True)
        a_img_stack = batch['a_img'].to(device, non_blocking=True)
        l_img_stack = batch['l_img'].to(device, non_blocking=True)
        la_img_stack = batch['la_img'].to(device, non_blocking=True)
        txt_inputs = batch['ids'].to(device, non_blocking=True)
        txt_attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        txt_labels = batch['labels'].to(device, non_blocking=True)

        # 维度检查和调整
        if len(img_stack.shape) == 4:
            img_stack = img_stack.unsqueeze(1)
            a_img_stack = a_img_stack.unsqueeze(1)
            l_img_stack = l_img_stack.unsqueeze(1)
            la_img_stack = la_img_stack.unsqueeze(1)

        if len(txt_inputs.shape) == 2:
            txt_inputs = txt_inputs.unsqueeze(1)
            txt_attention_mask = txt_attention_mask.unsqueeze(1)
            txt_labels = txt_labels.unsqueeze(1)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # 仅保留：流形对齐(融合全局+局部) + MLM + MIM
            (loss_align, loss_mlm, loss_mim, alignment_info) = model(
                img_stack, a_img_stack, l_img_stack, la_img_stack,
                txt_inputs, txt_attention_mask, txt_labels,
                mask_ratio=mask_ratio
            )

            # 🔥 从alignment_info提取辅助统计（用于监控）
            if alignment_info is not None:
                sim_map = alignment_info.get('similarity_map', None)
                weights = alignment_info.get('weights', None)

                if sim_map is not None:
                    epoch_similarities.append(sim_map.detach().cpu())
                if weights is not None:
                    epoch_weights.append(weights.detach().cpu())

            # ========== 权重调度(保持你原逻辑): ManifoldAlign + MLM + MIM ==========
            base_w_align = getattr(args, 'w_align', 1.0)
            base_w_mlm = getattr(args, 'w_mlm', 0.2)
            base_w_mim = getattr(args, 'w_mim', 1.0)

            if epoch < 50:
                w_align = 1.0 * base_w_align
                w_mlm = 1.0 * base_w_mlm
                w_mim = 1.0 * base_w_mim
            elif epoch < 60:
                progress = (epoch - 20) / 40.0
                w_align = 1.0 * base_w_align
                w_mlm = (1.0 + 0.5 * progress) * base_w_mlm
                w_mim = (1.0 - 0.3 * progress) * base_w_mim
            else:
                w_align = 1.0 * base_w_align
                w_mlm = 1.5 * base_w_mlm
                w_mim = 0.7 * base_w_mim

            loss_align = torch.clamp(loss_align, min=0.0)
            loss_mlm = torch.clamp(loss_mlm, min=0.0)
            loss_mim = torch.clamp(loss_mim, min=0.0)

            total_loss = (
                w_align * loss_align +
                w_mlm * loss_mlm +
                w_mim * loss_mim
            )
            # ============================================================ #

        if not math.isfinite(total_loss.item()):
            print(f"Loss is {total_loss.item()}, stopping training")
            print(f"  loss_align: {loss_align.item()}")
            print(f"  loss_mlm: {loss_mlm.item()}")
            print(f"  loss_mim: {loss_mim.item()}")
            sys.exit(1)

        total_loss_value = total_loss.item()
        loss_align_v = loss_align.item()
        loss_mlm_v = loss_mlm.item()
        loss_mim_v = loss_mim.item()

        # 梯度更新
        total_loss = total_loss / accum_iter
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        # 🔥 每100步打印对齐统计（原样保留）
        if data_iter_step % 100 == 0 and misc.is_main_process():
            if alignment_info is not None:
                sim_map = alignment_info.get('similarity_map', None)
                weights = alignment_info.get('weights', None)

                if sim_map is not None and weights is not None:
                    sim_mean = sim_map.mean().item()
                    sim_std = sim_map.std().item()
                    sim_max = sim_map.max().item()
                    sim_min = sim_map.min().item()

                    weight_mean = weights.mean().item()
                    weight_std = weights.std().item()
                    weight_max = weights.max().item()
                    weight_min = weights.min().item()

                    print(f"\n  📊 Alignment Stats @ Step {data_iter_step}:")
                    print(f"     Similarity: mean={sim_mean:.3f}, std={sim_std:.3f}, range=[{sim_min:.3f}, {sim_max:.3f}]")
                    print(f"     Weights:    mean={weight_mean:.3f}, std={weight_std:.3f}, range=[{weight_min:.3f}, {weight_max:.3f}]")
                    alpha_img = alignment_info.get("alpha_img", None)
                    alpha_txt = alignment_info.get("alpha_txt", None)
                    if alpha_img is not None and alpha_txt is not None:
                        print(f"     Fusion α:  img_mean={alpha_img.mean().item():.3f}, txt_mean={alpha_txt.mean().item():.3f}")

                    if sim_std < 0.10 or weight_std < 0.15:
                        print(f"     ⚠️  WARNING: Possible mode collapse detected!")
                        print(f"     - Similarity std too low: {sim_std:.3f} < 0.10")
                        print(f"     - Weight std too low: {weight_std:.3f} < 0.15")

        # 可视化（保持你原触发条件）
        if (data_iter_step % 1000 == 0 and
            epoch >= 0 and
            misc.is_main_process() and
            vis_dir is not None):

            try:
                # 1) 保留你原来的文件名 alignment_epochXXX_iterXXXX.png
                visualize_manifold_alignment(
                    model=model,
                    img_sample=img_stack[:1],
                    txt_inputs_sample=txt_inputs[:1],
                    txt_attention_mask_sample=txt_attention_mask[:1],
                    device=device,
                    save_dir=vis_dir,
                    epoch=epoch,
                    iteration=data_iter_step,
                    topk_ratio=getattr(args, "vis_topk_ratio", 0.30),
                )

                # 2) 额外保存两张“更顶刊解释性”的图（maps/tokens），同一时机保存
                if visualizer is not None:
                    visualizer.visualize_single_alignment_sample(
                        img=img_stack[:1],
                        txt_inputs=txt_inputs[:1],
                        txt_attention_mask=txt_attention_mask[:1],
                        epoch=epoch,
                        iteration=data_iter_step,
                        save_prefix="alignment",
                        topk_patches=getattr(args, "vis_topk_ratio", 0.30),
                        decode_tokens=True,
                    )

            except Exception as e:
                print(f"[WARNING] Visualization failed: {e}")

        # 更新指标
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(
            loss1=loss_align_v,
            loss2=loss_mlm_v,
            loss3=loss_mim_v,
            total_loss=total_loss_value
        )

        # TensorBoard记录
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss1_align", misc.all_reduce_mean(loss_align_v), epoch_1000x)
            log_writer.add_scalar("train/loss2_mlm", misc.all_reduce_mean(loss_mlm_v), epoch_1000x)
            log_writer.add_scalar("train/loss3_mim", misc.all_reduce_mean(loss_mim_v), epoch_1000x)
            log_writer.add_scalar("train/total_loss", misc.all_reduce_mean(total_loss_value), epoch_1000x)
            log_writer.add_scalar("train/lr", lr, epoch_1000x)

    # Epoch结束统计（原样保留）
    if misc.is_main_process() and len(epoch_similarities) > 0:
        all_sim = torch.cat(epoch_similarities, dim=0)
        all_weights = torch.cat(epoch_weights, dim=0)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Alignment Summary:")
        print(f"{'='*70}")
        print(f"  Similarity: mean={all_sim.mean():.3f}, std={all_sim.std():.3f}")
        print(f"  Weights:    mean={all_weights.mean():.3f}, std={all_weights.std():.3f}")

        if all_sim.std() < 0.12:
            print(f"  ⚠️  WARNING: Low similarity diversity (std={all_sim.std():.3f})")
        if all_weights.std() < 0.18:
            print(f"  ⚠️  WARNING: Low weight diversity (std={all_weights.std():.3f})")

        if all_sim.std() >= 0.15 and all_weights.std() >= 0.20:
            print(f"  ✅ Alignment looks healthy!")

        print(f"{'='*70}\n")

    # 每10个epoch可视化一次（原样保留）
    if misc.is_main_process() and (epoch + 1) % 5 == 0:
        try:
            visualizer_ep = ALTAVisualizer(
                model=model,
                device=device,
                save_dir=os.path.join(args.output_dir, 'visualizations')
            )
            visualizer_ep.visualize_fusion_weights(
                dataloader=data_loader,
                max_samples=200,
                epoch=epoch
            )
            print(f"[Epoch {epoch}] ✓ Fusion weights visualized")
        except Exception as e:
            print(f"[Warning] Visualization failed: {e}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _overlay_heatmap_on_image(img: np.ndarray, heat_2d: np.ndarray) -> np.ndarray:
    """Overlay patch-grid heatmap onto image. img can be HxW or HxWx3 in [0,1]."""
    h = heat_2d.astype(np.float32)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    if img.ndim == 2:
        H, W = img.shape
        base = np.stack([img, img, img], axis=-1)
    else:
        H, W = img.shape[:2]
        base = img

    gh, gw = h.shape
    scale_h = max(1, H // gh)
    scale_w = max(1, W // gw)
    up = np.kron(h, np.ones((scale_h, scale_w), dtype=np.float32))
    up = up[:H, :W]

    cmap = plt.get_cmap("jet")
    colored = cmap(up)[..., :3]

    alpha = 0.50
    out = np.clip((1 - alpha) * base + alpha * colored, 0.0, 1.0)
    return out


def _draw_topk_patch_boxes(ax, img: np.ndarray, weights_2d: np.ndarray, topk_ratio: float = 0.30) -> None:
    """Draw Top-K patch boxes on the image according to weights."""
    gh, gw = weights_2d.shape
    if img.ndim == 2:
        H, W = img.shape
    else:
        H, W = img.shape[:2]

    n = gh * gw
    k = max(1, int(n * float(topk_ratio)))
    flat = weights_2d.reshape(-1)
    top_idx = np.argsort(-flat)[:k]
    mask = np.zeros_like(flat, dtype=np.float32)
    mask[top_idx] = 1.0
    mask_2d = mask.reshape(gh, gw)

    ph = H / gh
    pw = W / gw

    for r in range(gh):
        for c in range(gw):
            if mask_2d[r, c] <= 0:
                continue
            rect = Rectangle(
                (c * pw, r * ph),
                pw, ph,
                fill=False,
                linewidth=1.2,
                edgecolor=(1.0, 0.2, 0.2, 0.70),
            )
            ax.add_patch(rect)


def visualize_manifold_alignment(model, img_sample, txt_inputs_sample,
                                txt_attention_mask_sample, device,
                                save_dir, epoch, iteration,
                                topk_ratio: float = 0.30):
    """
    可视化流形对齐效果（升级版，更可解释，但保持你的输出文件名不变）

    仍然保存： alignment_epoch{epoch:03d}_iter{iteration:04d}.png
    但从 1x3 升级为 1x5：
      (a) Pixel-Word Similarity
      (b) Alignment Weights
      (c) Weight Overlay on Image
      (d) Top-K patches on Image
      (e) Original Image + basic stats
    """
    model.eval()

    with torch.no_grad():
        target_model = model.module if hasattr(model, "module") else model

        # 图像编码（不mask，确保patch可见）
        img_global, img_local, _, _, _ = target_model.image_encoder(
            img_sample.squeeze(1) if len(img_sample.shape) == 5 else img_sample,
            mask_ratio=0.0
        )

        # 文本编码
        txt_in = txt_inputs_sample.squeeze(1) if len(txt_inputs_sample.shape) == 3 else txt_inputs_sample
        txt_mask = txt_attention_mask_sample.squeeze(1) if len(txt_attention_mask_sample.shape) == 3 else txt_attention_mask_sample

        txt_embed = target_model.bert_encoder(
            txt_in,
            txt_mask,
            output_cls_projected_embedding=True,
            return_dict=True
        ).cls_projected_embedding
        txt_embed = target_model.text_proj_adapter(txt_embed)
        txt_global = F.normalize(txt_embed, dim=-1)

        txt_output = target_model.bert_encoder.bert(
            input_ids=txt_in,
            attention_mask=txt_mask,
            return_dict=True
        )
        txt_hidden = txt_output.last_hidden_state
        txt_tokens = target_model.text_local_proj(txt_hidden[:, 1:, :])

        # 流形对齐信息
        _, alignment_info = target_model.manifold_align(
            img_global=img_global,
            img_patches=img_local,
            txt_global=txt_global,
            txt_tokens=txt_tokens,
            return_info=True
        )

        similarity_map = alignment_info['similarity_map'][0].cpu().numpy()  # [N]
        weights_map = alignment_info['weights'][0].cpu().numpy()            # [N]

        n_patches = similarity_map.shape[0]
        H = int(np.sqrt(n_patches))
        W = int(n_patches / H) if H > 0 else n_patches

        similarity_2d = similarity_map.reshape(H, W)
        weights_2d = weights_map.reshape(H, W)

        # 反归一化恢复图像显示
        img_np = None
        try:
            img_np = img_sample[0, 0].cpu().numpy()  # [C,H,W]
            if img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img_np[0]
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        except Exception:
            img_np = None

        # 画图：1x5
        fig, axes = plt.subplots(1, 5, figsize=(26, 5.5))

        # (a) similarity
        im1 = axes[0].imshow(similarity_2d, cmap='magma', interpolation='bilinear')
        axes[0].set_title(f'(a) Patch-Token Similarity\nEpoch {epoch}, Iter {iteration}')
        axes[0].set_xlabel('Patch X'); axes[0].set_ylabel('Patch Y')
        axes[0].grid(False)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # (b) weights
        im2 = axes[1].imshow(weights_2d, cmap='viridis', interpolation='bilinear', vmin=0, vmax=1)
        axes[1].set_title(f'(b) Alignment Weights\n(High = Strong Text Guidance)')
        axes[1].set_xlabel('Patch X'); axes[1].set_ylabel('Patch Y')
        axes[1].grid(False)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # (c) overlay
        if img_np is not None:
            overlay = _overlay_heatmap_on_image(img_np, weights_2d)
            axes[2].imshow(overlay, cmap='gray' if overlay.ndim == 2 else None)
            axes[2].set_title('(c) Weight Overlay on Image')
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, 'Overlay unavailable', ha='center', va='center')
            axes[2].set_title('(c) Overlay')
            axes[2].axis('off')

        # (d) top-k patches
        if img_np is not None:
            axes[3].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            _draw_topk_patch_boxes(axes[3], img_np, weights_2d, topk_ratio=topk_ratio)
            axes[3].set_title(f'(d) Top-{int(topk_ratio*100)}% Patches (by weight)')
            axes[3].axis('off')
        else:
            axes[3].text(0.5, 0.5, 'Top-K unavailable', ha='center', va='center')
            axes[3].set_title('(d) Top-K')
            axes[3].axis('off')

        # (e) original + stats
        if img_np is not None:
            axes[4].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            axes[4].set_title(
                f'(e) Original Image\n'
                f'AvgSim={similarity_map.mean():.3f}, StdSim={similarity_map.std():.3f}\n'
                f'AvgW={weights_map.mean():.3f}, StdW={weights_map.std():.3f}'
            )
            axes[4].axis('off')
        else:
            axes[4].text(
                0.5, 0.5,
                f'Image unavailable\n\n'
                f'AvgSim={similarity_map.mean():.3f}, StdSim={similarity_map.std():.3f}\n'
                f'AvgW={weights_map.mean():.3f}, StdW={weights_map.std():.3f}',
                ha='center', va='center'
            )
            axes[4].set_title('(e) Stats')
            axes[4].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'alignment_epoch{epoch:03d}_iter{iteration:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Visualization] Saved to: {save_path}")
        print(f"  Avg Pixel-Word Similarity: {similarity_map.mean():.4f}")
        print(f"  Avg Alignment Weight: {weights_map.mean():.4f}")

    model.train()


def compute_AUROCs(gt, pred):
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    return roc_auc_score(gt_np, pred_np)


def compute_acc(gt, pred):
    gt = gt.cpu().numpy().astype('bool')
    pred = pred.cpu().numpy().astype('bool')
    acc = np.mean(gt == pred).astype('float32')
    tp = np.sum(gt & pred).astype('float32')
    fp = np.sum(pred & ~gt).astype('float32')
    fn = np.sum(gt & ~pred).astype('float32')
    recall = tp / (tp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1 = 2 * prec * recall / (prec + recall + 1e-8)
    return acc, f1, recall, prec


@torch.no_grad()
def zeroshot_valid_one_epoch(model: torch.nn.Module,
                             data_loader: Iterable,
                             device: torch.device, epoch: int,
                             log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'
    print_freq = 20

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    pred_soft = torch.FloatTensor().to(device)

    validation_possible = False

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        try:
            with torch.cuda.amp.autocast():
                if isinstance(batch, dict):
                    if data_iter_step == 0:
                        print("[INFO] Validation dataloader returns dict format (train format).")
                        print("[INFO] Skipping zero-shot validation. Will only perform training.")
                    break
                elif len(batch) == 5:
                    _, target, sample, pos_batch_dict, neg_batch_dict = batch
                    validation_possible = True
                else:
                    if data_iter_step == 0:
                        print(f"[WARNING] Unexpected batch format: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
                        print("[INFO] Skipping zero-shot validation.")
                    break

                target = target.squeeze().to(device)
                target_model = model.module if hasattr(model, "module") else model

                img_feature = target_model.forward_image_feature(
                    sample.unsqueeze(1).to(device) if len(sample.shape) == 4 else sample.to(device),
                    mask_ratio=0.0
                )

                pos_text_feature = F.normalize(
                    target_model.bert_encoder(
                        pos_batch_dict['input_ids'].to(device),
                        pos_batch_dict['attention_mask'].to(device),
                        output_cls_projected_embedding=True,
                        return_dict=True
                    ).cls_projected_embedding,
                    dim=-1, p=2
                )

                neg_text_feature = F.normalize(
                    target_model.bert_encoder(
                        neg_batch_dict['input_ids'].to(device),
                        neg_batch_dict['attention_mask'].to(device),
                        output_cls_projected_embedding=True,
                        return_dict=True
                    ).cls_projected_embedding,
                    dim=-1, p=2
                )

                if hasattr(target_model, 'text_proj_adapter'):
                    pos_text_feature = target_model.text_proj_adapter(pos_text_feature)
                    neg_text_feature = target_model.text_proj_adapter(neg_text_feature)
                    pos_text_feature = F.normalize(pos_text_feature, dim=-1, p=2)
                    neg_text_feature = F.normalize(neg_text_feature, dim=-1, p=2)

                pos_cos_sim = (pos_text_feature * img_feature).sum(dim=1)
                neg_cos_sim = (neg_text_feature * img_feature).sum(dim=1)

                predict_soft = torch.softmax(
                    torch.cat([pos_cos_sim.unsqueeze(-1), neg_cos_sim.unsqueeze(-1)], dim=-1),
                    dim=-1
                )[:, 0]
                predict = pos_cos_sim > neg_cos_sim

                gt = torch.cat((gt, target.to(torch.int)), 0)
                pred = torch.cat((pred, predict.to(torch.int)), 0)
                pred_soft = torch.cat((pred_soft, predict_soft), 0)

        except Exception as e:
            if data_iter_step == 0:
                print(f"[WARNING] Validation failed with error: {e}")
                print("[INFO] This is normal if validation dataset format is different.")
                print("[INFO] Skipping zero-shot validation and returning dummy metrics.")
            break

    if validation_possible and len(gt) > 0:
        try:
            acc, f1, recall, prec = compute_acc(gt, pred)
            auroc = compute_AUROCs(gt, pred_soft)
            metric_logger.synchronize_between_processes()
            print("Validation stats:", metric_logger)
            print(f"AUROC: {auroc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Prec: {prec:.4f}")
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auroc, acc, f1, recall, prec
        except Exception as e:
            print(f"[WARNING] Failed to compute validation metrics: {e}")
            print("[INFO] Returning dummy metrics.")

    print("[INFO] Validation not performed. Returning dummy metrics (all zeros).")
    return {}, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
