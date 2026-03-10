import argparse
import copy
import os.path
import random
import time
from config.configs import Config
import logging
from logs.autolog import get_or_create_logger
import torch
import numpy as np
from models.build_model import load_d4_model
from detr.datasets import build_dataset, get_coco_api_from_dataset, CocoDetection
from torch.utils.data import DataLoader,BatchSampler,RandomSampler,SequentialSampler
import detr.util.misc as detr_utils
from utils.tools import auto_unit,auto_number_length
from pathlib import Path
from engine import train_one_epoch,evaluate
import json
import datetime
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo
)
from typing import Optional
import shutil

def wait_for_gpu_memory(
    required_free_mib: int,
    gpu_index: int = 0,
    check_interval_s: float = 60,
    timeout_s: Optional[float] = None,
) -> None:
    """
    阻塞等待直到指定 GPU 的空闲显存 >= required_free_mib。
    - required_free_mib: 需要的空闲显存（MiB）
    - gpu_index: GPU 编号（nvidia-smi 里的 index）
    - check_interval_s: 检查间隔
    - timeout_s: 超时秒数；None 表示永不超时
    """
    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        start = time.time()

        while True:
            info = nvmlDeviceGetMemoryInfo(handle)
            free_mib = info.free // (1024 * 1024)
            total_mib = info.total // (1024 * 1024)
            used_mib = info.used // (1024 * 1024)

            if free_mib >= required_free_mib:
                # 满足条件：继续执行
                print(f"[OK] GPU{gpu_index} free={free_mib}MiB (need {required_free_mib}MiB), "
                      f"used={used_mib}/{total_mib}MiB")
                return

            # 不满足：等待
            print(f"[WAIT] GPU{gpu_index} free={free_mib}MiB < {required_free_mib}MiB, "
                  f"used={used_mib}/{total_mib}MiB, sleep {check_interval_s}s")
            time.sleep(check_interval_s)

            if timeout_s is not None and (time.time() - start) >= timeout_s:
                raise TimeoutError(
                    f"Timeout: GPU{gpu_index} free memory not reached {required_free_mib}MiB in {timeout_s}s"
                )
    finally:
        nvmlShutdown()


def get_args_parser():
    parser = argparse.ArgumentParser('Dino Deformable DETR Detector(d4) Model Trainer')
    parser.add_argument('--cfg', type=str,required=True)
    parser.add_argument('--logger_dir', default='./logs', type=str)
    parser.add_argument('--logger_level', default='info', type=str)
    parser.add_argument('--logger_filename', default='', type=str)

    return parser

def count_dataset_info(dataset: CocoDetection):
    coco = dataset.coco
    num_images = len(dataset)
    num_annotations = len(coco.anns)
    num_categories = len(coco.cats)
    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {cat['id']: cat['name'] for cat in cats}
    # 统计每个类别的标注数量
    cat_id_to_count = {}
    for cat_id in coco.getCatIds():
        ann_ids = coco.getAnnIds(catIds=cat_id)
        cat_id_to_count[cat_id] = len(ann_ids)
        # print(f"{id_to_name[cat_id]}: {len(ann_ids)}")
    cat_id_to_label = dataset.cat_id_to_label
    return {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'num_categories': num_categories,
        'cat_id_to_name':id_to_name,
        'cat_id_to_label':cat_id_to_label,
        'cat_id_to_count':cat_id_to_count,
    }

def check_last_num(arr:list, l:int):
    if not isinstance(l,int) or l <  2:
        if l == -1:
            l = len(arr)
        else:
            raise ValueError(f'l必须是大于2的整数(-1除外),现在l为{l}.')
    if len(arr)<l:
        return False
    last_num = arr[-l:]
    return all(last_num[i]>last_num[i+1] for i in range(l-1))

def check_early_exist(arr:list,l):
    flag = check_last_num(arr, l)
    if flag:
        if arr[0]-arr[-1]>0.05:
            return True
        return False

def test_stats2text(stats_dict:dict,epoch=0):
    test_stats = copy.deepcopy(stats_dict)
    coco_eval_bbox = test_stats.get("coco_eval_bbox")
    del test_stats["coco_eval_bbox"]
    res = f"Epoch[{epoch}]:\n"
    for k,v in test_stats.items():
        res += f" -{k}: {v}\n"
    key_map = {
        0 : "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        1 : "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
        2 : "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
        3 : "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        4 : "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        5 : "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
        6: "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
        7: "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
        8: "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        9: "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        10: "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        11: "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
    }
    res += " -coco_eval_bbox:\n"
    for i,v in enumerate(coco_eval_bbox):
        res += f"  |-{key_map[i]}={v}\n"
    return res

def main(cfg:Config,*,logger:logging.Logger|None=None):
    wait_for_gpu_memory(30000,check_interval_s=300)
    if logger is None:
        if cfg.eval:
            logger = get_or_create_logger(r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/logs',
                                          "d4_trainer",
                                          logging.INFO)
        else:
            logger = get_or_create_logger(r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/logs',
                                      "evaluator",
                                      logging.INFO)
    logger.info(f"训练脚本已启动")
    device = torch.device(cfg.device)
    logger.info(f"本次训练使用硬件:{cfg.device}")
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"开始构建模型系统")
    model, criterion, postprocessors, _ = load_d4_model(cfg,cfg.model_path,logger=logger,is_train=True) # 构建模型系统
    logger.info(f"模型系统构建成功")
    if cfg.model_path is not None:
        logger.info(f"模型加载权重自{os.path.abspath(cfg.model_path)}")
    logger.info(f"开始构建数据集")
    dataset_train = build_dataset(image_set='train', args=cfg)
    count_result = count_dataset_info(dataset_train)
    string_map=[]
    for cid,n in count_result['cat_id_to_name'].items():
        string_map.append((cid,n,count_result['cat_id_to_label'][cid],count_result['cat_id_to_count'][cid]))
    string_all = f"""训练集加载完毕:
-图片数:{count_result.get('num_images','null')}
-标注数:{count_result.get('num_annotations','null')}
-类别数:{count_result.get('num_categories','null')}
-类别详情(格式为(CategoriesId,name,label,类标注数)):{string_map}
"""
    logger.info(string_all)
    dataset_val = build_dataset(image_set='val', args=cfg)
    count_result = count_dataset_info(dataset_val)
    string_map = []
    for cid, n in count_result['cat_id_to_name'].items():
        string_map.append((cid, n, count_result['cat_id_to_label'][cid], count_result['cat_id_to_count'][cid]))
    string_all = f"""验证集加载完毕:
-图片数:{count_result.get('num_images', 'null')}
-标注数:{count_result.get('num_annotations', 'null')}
-类别数:{count_result.get('num_categories', 'null')}
-类别详情(格式为(CategoriesId,name,label,类标注数)):{string_map}
    """
    logger.info(string_all)
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)

    batch_sampler_train = BatchSampler(
        sampler_train, cfg.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=detr_utils.collate_fn, num_workers=cfg.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, cfg.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=detr_utils.collate_fn, num_workers=cfg.num_workers,
                                 pin_memory=True)
    def match_name_keywords(n, name_keywords):
        if name_keywords is None or n is None:return False
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
    all_para_name = []
    all_para_name_size = []
    all_para_name_shape = []
    for n, p in model.named_parameters():
        all_para_name.append(n)
        all_para_name_size.append(p.numel())
        all_para_name_shape.append(p.shape)
    f_string = ""
    for n,s,ss in zip(all_para_name, all_para_name_size,all_para_name_shape):
        f_string += f"\n{n}:{s}({ss})"
    logger.info(f'D4模型所有参数及其尺寸:{f_string}')
    # 参数分组
    backbone_params = []
    proj_params = []
    other_params = []
    for n,p in model.named_parameters():
        if p.requires_grad:
            if match_name_keywords(n,cfg.lr_backbone_names):
                backbone_params.append(p)
            elif match_name_keywords(n,cfg.lr_linear_proj_names):
                proj_params.append(p)
            else:
                other_params.append(p)
    param_dicts = [
        {
            "params":other_params,
            "lr": cfg.lr,
        },
        {
            "params": backbone_params,
            "lr": cfg.lr_backbone,
        },
        {
            "params": proj_params,
            "lr": cfg.lr * cfg.lr_linear_proj_mult,
        }
    ]

    if cfg.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.lr, momentum=0.9,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop) # 学习率调度器

    base_ds = get_coco_api_from_dataset(dataset_val) # COCO评估系统
    output_dir = Path(cfg.output_dir)
    if cfg.resume:
        checkpoint = torch.load(cfg.resume,map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not cfg.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            logger.info(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if cfg.override_resumed_lr_drop:
            logger.warning('警告:override_resumed_lr_drop被设置为True,系统将不会完全继承原训练的学习率调度策略。')
            lr_scheduler.step_size = cfg.lr_drop # 覆盖衰减步长
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups)) # 重置基准学习率
        lr_scheduler.step(lr_scheduler.last_epoch)
        cfg.start_epoch = checkpoint['epoch'] + 1
        if not cfg.eval: # 检查继承的模型是否正常
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds,
                device, cfg.output_dir,external_logger=logger
            )
    if cfg.eval: # 评价模式
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds,
            device, cfg.output_dir,external_logger=logger
        )
        if cfg.output_dir:
            detr_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    if cfg.early_exit_enable:
        if cfg.early_exit_threshold:
            eet = cfg.early_exit_threshold
        else:
            if cfg.epochs<=200:
                eet = int(cfg.epochs * 0.1) + 1
            else:
                eet = 25
        min_train_epoch = int(cfg.min_train_epoch*cfg.epochs) if isinstance(cfg.min_train_epoch,float) else int(cfg.min_train_epoch)
    logger.info("开始训练")
    start_time = time.time()
    best_checkpoint_map = 0
    best_checkpoint_map_epoch = 0
    # 训练主循环
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) # 获取未冻结的总参数量
    for epoch in range(cfg.start_epoch, cfg.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, cfg.clip_max_norm,external_logger=logger)
        lr_scheduler.step()
        if cfg.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # lr_drop输出一次且每5个epoch输出一次
            if (epoch + 1) % cfg.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{auto_number_length(epoch,max_length_num=cfg.epochs)}.pth')
            for checkpoint_path in checkpoint_paths:
                detr_utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': cfg,
                },checkpoint_path)
        # 验证
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.output_dir,external_logger=logger
        )
        if best_checkpoint_map < test_stats['coco_eval_bbox'][0]:
            detr_utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': cfg,
            }, Path(cfg.output_dir) / 'best.pth')
            best_checkpoint_map = test_stats['coco_eval_bbox'][0]
            best_checkpoint_map_epoch = epoch
        if cfg.early_exit_enable:
            cfg.early_cache.append(test_stats['coco_eval_bbox'][0])
            if len(cfg.early_cache) > eet:
                cfg.early_cache = cfg.early_cache[-eet:]
                if epoch >= min_train_epoch:
                    cfg.early_exit = check_early_exist(cfg.early_cache,eet)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if cfg.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            def mix_dict(_dict1,_dict2):
                new_dict = {}
                for k,v in _dict1.items():
                    if k not in ("epoch","step"):
                        new_dict[k] = _dict1[k]
                for k,v in _dict2.items():
                    if k not in ("epoch", "step"):
                        new_dict[k] = _dict2[k]
                return new_dict
            logger.info(f"训练信息:\n{log_stats}",extra={"metrics":log_stats,"epoch":epoch})
            logger.info(f" -其中验证集评价结果:\n{test_stats2text(test_stats,epoch)}",)
            logger.info(f"Epoch[{epoch}]mAP: {test_stats['coco_eval_bbox'][0]:.4f}")
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 10 == 0:
                        filenames.append(f'{auto_number_length(epoch,max_length_num=cfg.epochs)}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
        if cfg.early_exit:
            logger.info(f"检测到验证mAP连续{eet}轮下降，系统提前退出。")
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"训练完毕,总耗时：{total_time_str}.最高mAP为{best_checkpoint_map}(epoch[{best_checkpoint_map_epoch}])")

def train2map(target_map,max_epoch,*,cfg:Config,logger:logging.Logger|None):
    wait_for_gpu_memory(30000,check_interval_s=300)
    if logger is None:
        logger = get_or_create_logger(r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/logs',
                                          "d4_trainer",
                                          logging.INFO)
    logger.info(f"训练脚本已启动")
    device = torch.device(cfg.device)
    logger.info(f"本次训练使用硬件:{cfg.device}")
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"开始构建模型系统")
    model, criterion, postprocessors, _ = load_d4_model(cfg, cfg.model_path, logger=logger, is_train=True)  # 构建模型系统
    logger.info(f"模型系统构建成功")
    if cfg.model_path is not None:
        logger.info(f"模型加载权重自{os.path.abspath(cfg.model_path)}")
    logger.info(f"开始构建数据集")
    dataset_train = build_dataset(image_set='train', args=cfg)
    count_result = count_dataset_info(dataset_train)
    string_map = []
    for cid, n in count_result['cat_id_to_name'].items():
        string_map.append((cid, n, count_result['cat_id_to_label'][cid], count_result['cat_id_to_count'][cid]))
    string_all = f"""训练集加载完毕:
    -图片数:{count_result.get('num_images', 'null')}
    -标注数:{count_result.get('num_annotations', 'null')}
    -类别数:{count_result.get('num_categories', 'null')}
    -类别详情(格式为(CategoriesId,name,label,类标注数)):{string_map}
    """
    logger.info(string_all)
    dataset_val = build_dataset(image_set='val', args=cfg)
    count_result = count_dataset_info(dataset_val)
    string_map = []
    for cid, n in count_result['cat_id_to_name'].items():
        string_map.append((cid, n, count_result['cat_id_to_label'][cid], count_result['cat_id_to_count'][cid]))
    string_all = f"""验证集加载完毕:
    -图片数:{count_result.get('num_images', 'null')}
    -标注数:{count_result.get('num_annotations', 'null')}
    -类别数:{count_result.get('num_categories', 'null')}
    -类别详情(格式为(CategoriesId,name,label,类标注数)):{string_map}
        """
    logger.info(string_all)
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)

    batch_sampler_train = BatchSampler(
        sampler_train, cfg.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=detr_utils.collate_fn, num_workers=cfg.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, cfg.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=detr_utils.collate_fn, num_workers=cfg.num_workers,
                                 pin_memory=True)

def copy_file(src_path: str, dst_dir: str, name: str | None = None) -> str:
    src = Path(src_path)
    dst_root = Path(dst_dir)

    if name is None:
        name = src.name
    else:
        # 只取文件名，避免 name 里带路径（可按你的需求删掉这一行）
        name = Path(name).name

    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / name

    shutil.copy2(src, dst)
    return str(dst)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    lv = logging.getLevelName(args.logger_level.upper())
    if args.logger_filename:
        f = args.logger_filename
    else:
        f = None
    logger = get_or_create_logger(args.logger_dir,"d4_trainer",lv, f)
    cfg = Config(args.cfg)
    main(cfg, logger=logger)
