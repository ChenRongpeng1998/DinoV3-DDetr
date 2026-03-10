import math
import os
import sys
from typing import Iterable
from logs.autolog import get_or_create_logger

import torch
import detr.util.misc as detr_utils
from detr.datasets.coco_eval import CocoEvaluator
from detr.datasets.data_prefetcher import data_prefetcher
from logs.autolog import get_or_create_logger
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,*,
                    external_logger=None):
    model.train()
    criterion.train()
    # 初始化指标记录器及日志记录器
    logger = get_or_create_logger(name='trainer',external_logger=external_logger)
    metric_logger = detr_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', detr_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', detr_utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', detr_utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # 数据预取器初始化
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        # 损失异常检查
        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training\n{loss_dict}")
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        # 梯度裁剪
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = detr_utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step() # 参数更新
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # 预取数据
        samples, targets = prefetcher.next()
    logger.info(f'Averaged stats:\n{metric_logger}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,*,external_logger=None):
    model.eval()
    criterion.eval()
    logger = get_or_create_logger(name="evaluator",external_logger=external_logger) # 日志记录器
    # 初始化指标记录器
    metric_logger = detr_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', detr_utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    coco_evaluator = CocoEvaluator(base_ds, ['bbox']) # 初始化COCO 评估器
    # 主循环
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # 移动数据到指定设备
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples) # 前向推理
        # 计算损失（用于监控，不影响评估指标）
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # 计算加权总损失
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        metric_logger.update(
            loss=loss.item(),  # 总损失
            loss_ce=loss_dict['loss_ce'].item() * weight_dict['loss_ce'],
            loss_bbox=loss_dict['loss_bbox'].item() * weight_dict['loss_bbox'],
            loss_giou=loss_dict['loss_giou'].item() * weight_dict['loss_giou'],
            class_error=loss_dict['class_error'].item()
        )
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)} # 构建 {image_id: prediction} 映射
        if coco_evaluator is not None:
            coco_evaluator.update(res)
    logger.info(f"Averaged stats:\n{metric_logger}")
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    # 整理结果
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        # logger.info(f"mAP: {stats['coco_eval_bbox'][0]:.4f}")
    return stats, coco_evaluator
