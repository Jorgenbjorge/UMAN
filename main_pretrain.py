# ========== main_pretrain.py (修复版) ==========

import argparse
import datetime
import json
import shutil
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import timm

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import model as alta_model
from engine_pretrain import train_one_epoch
import pretrain_datasets
from busi_eval_utils import evaluate_zero_shot_busi


# ========== 🔥 修复：手动实现 add_weight_decay 函数 ==========
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    为模型参数添加权重衰减，跳过某些特定层（如bias, LayerNorm等）
    
    Args:
        model: PyTorch模型
        weight_decay: 权重衰减系数
        skip_list: 要跳过的参数名列表
    
    Returns:
        param_groups: 优化器参数组列表
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过冻结的参数
        
        # 判断是否应该跳过权重衰减
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            # 1D参数（bias, LayerNorm weights等）不使用权重衰减
            no_decay.append(param)
        else:
            # 2D+参数（卷积、全连接权重等）使用权重衰减
            decay.append(param)
    
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]
# ================================================================


# ========== 参数解析部分 ==========

def get_args_parser():
    parser = argparse.ArgumentParser('ALTA pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='alta', type=str)
    parser.add_argument('--proj_dim', default=512, type=int)
    parser.add_argument('--adapter_type', default='normal', type=str)
    parser.add_argument('--adapter_dim', default=256, type=int)
    parser.add_argument('--adapter_rate', default=0.5, type=float)
    parser.add_argument('--adapter_mlp_ratio', default=0.25, type=float)
    parser.add_argument('--adapter_t_range', default=5, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1.5e-4)
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--data_path', default='/media/profz/data1/hmd/finetune/', type=str)
    parser.add_argument('--image_dirs', nargs='+', default=['images', 'images2', 'images3'])
    parser.add_argument('--output_dir', default='/media/profz/data1/hmd/MTG_new/v2_2/no_topo')
    parser.add_argument('--log_dir', default='/media/profz/data1/hmd/MTG_new/v2_2/no_topo/logs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--from_begin', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--clip_temp', default=0.07, type=float)
    parser.add_argument('--loss_weight', default=0.5, type=float)
    parser.add_argument('--mae_path', default='/media/profz/data1/hmd/ALTA/vision_encoder_weights/MRM.pth', type=str)
    parser.add_argument('--bert_path', default='/media/profz/data1/hmd/Bio_ClinicalBERT', type=str)
    parser.add_argument('--note', default='', type=str)
    parser.add_argument('--script', default='', type=str)
    parser.add_argument('--dist_eval', action='store_true')

    # === 新增参数 ===
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Frequency of zero-shot evaluation (in epochs)')
    parser.add_argument('--keep_only_best', action='store_true',
                        help='Only keep best models, delete periodic checkpoints to save disk')
    parser.add_argument('--w_align', type=float, default=1.0)
    parser.add_argument('--w_mlm', type=float, default=0.2)
    parser.add_argument('--w_mim', type=float, default=1.0)
    parser.add_argument('--ablation_mode', type=str, default='')
    parser.add_argument('--align_topo_w', type=float, default=0.20)
    
    # 🔥 Early Stopping 参数
    parser.add_argument('--patience', default=15, type=int,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min_delta', default=0.001, type=float,
                        help='Minimum improvement to reset patience counter')
    parser.add_argument('--lr_patience', default=5, type=int,
                        help='Reduce LR after this many epochs without improvement')
    parser.add_argument('--lr_factor', default=0.5, type=float,
                        help='Learning rate reduction factor')

    return parser


# ========== 主函数部分 ==========

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 固定随机种子
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset_train = pretrain_datasets.ALTADataset(
        data_root=args.data_path, is_train=True, args=args)
    print(f"Total number of training samples: {len(dataset_train)}")
    print(dataset_train.get_stats())

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    log_writer = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )

    model = alta_model.ALTA_ViT(args=args)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    # 🔥 修复：使用自定义的 add_weight_decay 函数
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # === 🔥 初始化Early Stopping相关变量 ===
    best_auc = 0.0
    best_acc = 0.0
    best_combined = 0.0
    best_epoch = 0
    
    patience_counter = 0  # 未提升的epoch计数
    lr_patience_counter = 0  # 学习率调整计数
    
    # 用于存储历史最优指标
    history_metrics = {
        'best_auc': best_auc,
        'best_acc': best_acc,
        'best_combined': best_combined,
        'best_epoch': best_epoch,
    }

    # === 训练循环 ===
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 🔥 训练一个epoch
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device,
            epoch, loss_scaler, log_writer=log_writer, args=args
        )

        # === 🔥 Zero-shot评估 ===
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            print(f"\n{'='*60}")
            print(f"Running Zero-shot Evaluation at Epoch {epoch}")
            print(f"{'='*60}")

            metrics = evaluate_zero_shot_busi(model, args, device, epoch, args.output_dir)
            acc, auc = metrics['accuracy'], metrics['auc']
            f1 = metrics['f1']
            combined = acc + auc  # 组合指标
            
            # 🔥 判断是否有提升
            improved = False
            improvement_msg = []
            
            # 检查各个指标是否提升
            if auc > history_metrics['best_auc'] + args.min_delta:
                history_metrics['best_auc'] = auc
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler,
                                epoch=epoch, name="best_auc")
                improvement_msg.append(f"AUC: {auc:.4f} (↑)")
                improved = True

            if acc > history_metrics['best_acc'] + args.min_delta:
                history_metrics['best_acc'] = acc
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler,
                                epoch=epoch, name="best_acc")
                improvement_msg.append(f"Acc: {acc*100:.2f}% (↑)")
                improved = True

            if combined > history_metrics['best_combined'] + args.min_delta:
                history_metrics['best_combined'] = combined
                history_metrics['best_epoch'] = epoch
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler,
                                epoch=epoch, name="best_combined")
                improvement_msg.append(f"Combined: {combined:.4f} (↑)")
                improved = True
            
            # 🔥 根据是否提升，更新patience计数器
            if improved:
                patience_counter = 0
                lr_patience_counter = 0
                print(f"\n✅ Model Improved! {', '.join(improvement_msg)}")
            else:
                patience_counter += 1
                lr_patience_counter += 1
                print(f"\n⚠️  No improvement for {patience_counter} epoch(s)")
                print(f"   Current: Acc={acc*100:.2f}%, AUC={auc:.4f}, Combined={combined:.4f}")
                print(f"   Best: Acc={history_metrics['best_acc']*100:.2f}%, AUC={history_metrics['best_auc']:.4f}, Combined={history_metrics['best_combined']:.4f} @ Epoch {history_metrics['best_epoch']}")
            
            # 🔥 学习率衰减策略
            if lr_patience_counter >= args.lr_patience:
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = old_lr * args.lr_factor
                
                # 确保不低于min_lr
                if new_lr >= args.min_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"\n📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
                    lr_patience_counter = 0  # 重置lr计数器
                else:
                    print(f"\n⚠️  Learning rate already at minimum ({args.min_lr:.2e})")
            
            # 🔥 Early Stopping检查
            if patience_counter >= args.patience:
                print(f"\n{'='*60}")
                print(f"🛑 Early Stopping Triggered!")
                print(f"{'='*60}")
                print(f"No improvement for {args.patience} consecutive epochs.")
                print(f"Best Combined Score: {history_metrics['best_combined']:.4f} @ Epoch {history_metrics['best_epoch']}")
                print(f"  - Best AUC: {history_metrics['best_auc']:.4f}")
                print(f"  - Best Acc: {history_metrics['best_acc']*100:.2f}%")
                break
            
            # 记录到TensorBoard
            if log_writer:
                log_writer.add_scalar('zeroshot/accuracy', acc, epoch)
                log_writer.add_scalar('zeroshot/auc', auc, epoch)
                log_writer.add_scalar('zeroshot/f1', f1, epoch)
                log_writer.add_scalar('zeroshot/combined', combined, epoch)
                log_writer.add_scalar('zeroshot/best_combined', history_metrics['best_combined'], epoch)
                log_writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # === 周期性保存checkpoint ===
        if args.output_dir and (epoch + 1) % args.save_freq == 0:
            # 如果启用keep_only_best，只保存最优模型
            if not args.keep_only_best:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler,
                                epoch=epoch, name=f"checkpoint-{epoch:04d}")

        # 记录训练日志
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'patience_counter': patience_counter,
        }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # === 训练结束总结 ===
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Best Combined Score: {history_metrics['best_combined']:.4f} @ Epoch {history_metrics['best_epoch']}")
    print(f"  - Best AUC: {history_metrics['best_auc']:.4f}")
    print(f"  - Best Accuracy: {history_metrics['best_acc']*100:.2f}%")
    print(f"\nBest models saved as:")
    print(f"  - best_auc.pth (AUC={history_metrics['best_auc']:.4f})")
    print(f"  - best_acc.pth (Acc={history_metrics['best_acc']*100:.2f}%)")
    print(f"  - best_combined.pth (Combined={history_metrics['best_combined']:.4f})")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTraining time: {total_time_str}')
    
    # 🔥 保存最终的训练总结
    summary = {
        'total_epochs': epoch + 1,
        'best_epoch': history_metrics['best_epoch'],
        'best_auc': float(history_metrics['best_auc']),
        'best_acc': float(history_metrics['best_acc']),
        'best_combined': float(history_metrics['best_combined']),
        'training_time': total_time_str,
        'early_stopped': patience_counter >= args.patience,
    }
    
    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nTraining summary saved to: {os.path.join(args.output_dir, 'training_summary.json')}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)