import json
import os.path
import time
from pathlib import Path
from detr.datasets.coco import CocoDetection
import detr.datasets.transforms as T
from detr.datasets import get_coco_api_from_dataset
from main import count_dataset_info
from logs.autolog import get_or_create_logger
import logging
from models.build_model import load_d4_model
from config.configs import Config
from engine import evaluate
from torch.utils.data import DataLoader,SequentialSampler
import detr.util.misc as detr_utils
from main import wait_for_gpu_memory
from PIL import Image,ImageDraw, ImageFont
import torch
from torchvision.transforms import v2
import random

def xywh2xyxy(coor):
    x, y, w, h = coor
    x2 = x + w
    y2 = y + h
    return [x, y, x2, y2]


def make_transforms(*,normalize:None|T.Compose=None):
    if normalize is None:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return T.Compose([
        T.RandomResize([1024], max_size=1024),
        normalize,
    ])
def build_dataset(img_folder,anno_path,*,return_masks=False,transforms=None,cache_mode=False):
    return CocoDetection(img_folder,anno_path,transforms=transforms,return_masks=return_masks,cache_mode=cache_mode)


def eval_coco_dataset(img_folder,anno_path,outputs=None,*,cfg:Config,logger:logging.Logger|None=None):
    if logger is None:
        logger = get_or_create_logger(r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/logs',
                                      "evaluator",
                                      logging.INFO)
    # 建立并统计数据集
    ds = build_dataset(img_folder,anno_path,transforms=make_transforms())
    sampler = SequentialSampler(ds)
    ds_loader = DataLoader(ds,3, sampler=sampler,drop_last=False,
                           collate_fn=detr_utils.collate_fn, num_workers=cfg.num_workers,
                           pin_memory=True)
    count_result = count_dataset_info(ds)
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
    logger.info(f"开始构建模型系统")
    model, criterion, postprocessors, _ = load_d4_model(cfg,cfg.model_path, logger=logger)  # 构建模型系统
    base_ds = get_coco_api_from_dataset(ds)  # COCO评估系统
    test_stats, coco_evaluator = evaluate(
        model, criterion, postprocessors, ds_loader, base_ds, cfg.device, cfg.output_dir, external_logger=logger
    )
    print(test_stats)

def make_transform(resize_size: int = 256):
    """创建图像预处理变换"""
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def infer_one_img(img_path,outputs_dir=None,outputs_name=None,*,cfg:Config,logger:logging.Logger|None=None,
                  transformers=None,score_threshold=0.1,model=None,criterion=None,postprocessors=None,
                  label_to_name = ("pointer_instrument","wrench_valve","knob","button","tag","digital_instrument","disc_valve")):
    if logger is None:
        logger = get_or_create_logger(r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/logs',
                                      "infer_logger",
                                      logging.INFO)
    if transformers is None:
        transforms = make_transform(1024)
    image_pil = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image_pil.size  # PIL: (W,H)
    device = cfg.device
    input_tensor = transforms(image_pil).unsqueeze(0)  # 添加batch维度
    input_tensor = input_tensor.to(device)
    if model is None or criterion is None or postprocessors is None:
        logger.info(f"开始构建模型系统")
        model, criterion, postprocessors, _ = load_d4_model(cfg, cfg.model_path, logger=logger)  # 构建模型系统
    with torch.no_grad():
        outputs = model(input_tensor)
        target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
        results = postprocessors["bbox"](outputs, target_sizes)
        pred = results[0]
        scores = pred["scores"]  # [N]
        labels = pred["labels"]  # [N]
        boxes = pred["boxes"]  # [N,4] 绝对坐标 xyxy，已经对应原图尺寸

        # 阈值过滤
        keep = scores >= score_threshold
        scores = scores[keep].detach().cpu()
        labels = labels[keep].detach().cpu()
        boxes = boxes[keep].detach().cpu()
    readable = []
    for s, l, b in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        x1, y1, x2, y2 = b
        readable.append({
            "label": int(l),
            "score": float(s),
            "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        })
    # 可视化（画框）
    draw = ImageDraw.Draw(image_pil)
    for obj in readable:
        x1, y1, x2, y2 = obj["box_xyxy"]
        draw.rectangle([x1, y1, x2, y2], width=3)
        name = label_to_name[obj["label"]]
        draw.text((x1, max(0, y1 - 12)), f"{name} {obj['score']:.2f}")
    os.makedirs(outputs_dir, exist_ok=True)
    if outputs_name is None:
        if outputs_dir is None:
            save_path = './'+img_path.split('/')[-1].split('.')[0]+'_pred.jpg'
        else:
            save_path = outputs_dir + '/' + img_path.split('/')[-1].split('.')[0] + '_pred.jpg'
    else:
        if outputs_name.endswith('.jpg') or outputs_name.endswith('png'):
            if outputs_dir is None:
                save_path = './' + outputs_name
            else:
                save_path = outputs_dir + '/' + outputs_name
        else:
            if outputs_dir is None:
                save_path = './' + outputs_name.split('.')[0] + '.jpg'
            else:
                save_path = outputs_dir + '/' + outputs_name.split('.')[0] + '.jpg'
    image_pil.save(save_path)
    return readable,save_path


def eval_and_via(img_folder,cfg:Config,)


if __name__ == "__main__":
    cfg_path = "/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/config/train.yaml"
    img_folder = r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/data/LC2/val2017'
    anno_path = r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/data/LC2/annotations/instances_val2017.json'
    cfg = Config(cfg_path)
    # image_folder = '/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/data/LC2/val2017'
    # images = []
    # for img_name in os.listdir(img_folder):
    #     if img_name.endswith('.jpg') or img_name.endswith('.png'):
    #         images.append(os.path.join(img_folder, img_name))
    # n = min(50, len(images))
    # sample_50 = random.sample(images, n)
    # data = json.load(open(anno_path))
    # map_name2id = {}
    # for d in data["images"]:
    #     map_name2id[d.get("file_name")] = d["id"]
    # map_img_id2index = {}
    # for i,anno in enumerate(data["annotations"]):
    #     if map_img_id2index.get(anno["image_id"]) is None:
    #         map_img_id2index[anno["image_id"]] = [i]
    #     else:
    #         map_img_id2index[anno["image_id"]].append(i)
    # model, criterion, postprocessors, _ = load_d4_model(cfg, cfg.model_path)  # 构建模型系统
    # rs = {}
    # for i,img_path in enumerate(sample_50):
    #     print(f"{i+1}/{len(sample_50)}: {img_path}")
    #     r,_ = infer_one_img(img_path,'./pred',cfg=cfg,model=model, criterion=criterion, postprocessors=postprocessors)
    #     rs[img_path] = r
    # results = {}
    # for img_path in sample_50:
    #     img_name = os.path.basename(img_path)
    #     if map_name2id.get(img_name) is not None:
    #         img_id = map_name2id[img_name]
    #     else:
    #         continue
    #     anno_indexes = map_img_id2index[img_id]
    #     bboxes = []
    #     for index in anno_indexes:
    #         xyxy = xywh2xyxy(data["annotations"][index]["bbox"])
    #         l = data["annotations"][index]["category_id"] - 1
    #         bboxes.append({'label':l,'box_xyxy':xyxy})
    #     results[img_path] = {
    #         'pred':rs[img_path],
    #         "gt":bboxes,
    #     }
    # print(results)
    wait_for_gpu_memory(25000,check_interval_s=300)
    cfg.model_path = r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/outputs/backup/exp_1_map_20.pth'
    if cfg.model_path is not None:
        eval_coco_dataset(img_folder,anno_path,cfg=cfg)
    # infer_one_img(img_path,cfg=cfg)







