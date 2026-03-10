import logging
import torch
import torch.nn as nn
from config.configs import Config
from models.backbone_convnext import build_backbone
from detr.models.matcher import HungarianMatcher
from detr.models.deformable_detr import SetCriterion,MLP,PostProcess
from detr.models.deformable_transformer import build_deforamble_transformer
from detr.util.misc import inverse_sigmoid
from torchvision.transforms import v2
import math
import copy
from detr.util.misc import NestedTensor
import torch.nn.functional as F
from logs.autolog import get_or_create_logger
from utils.tools import auto_unit

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class D4Model(nn.Module):
    def __init__(self, backbone,transformer,num_classes,num_queries,num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        super().__init__()

        self.num_queries = num_queries
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.layers_to_use)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.class_embed = nn.Linear(hidden_dim, num_classes)  # 构建分类头
        # 初始化分类头偏置项
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 2)  # 构建回归头
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # 初始化投影层权重
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers # 计算预测头数量
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor|torch.Tensor):
        # 输入检查
        if isinstance(samples, NestedTensor):
            if not isinstance(samples.tensors,torch.Tensor):
                raise TypeError(f'samples.tensor must be a torch.Tensor,but now it is a {type(samples.tensors)}')
        elif isinstance(samples, torch.Tensor):
            samples = NestedTensor(samples, torch.zeros(samples.shape[0], samples.shape[2], samples.shape[3],
                      dtype=torch.bool, device=samples.device))
        else:
            raise TypeError(f'samples must be a NestedTensor or torch.Tensor,but now it is a {type(samples)}')
        features,pos = self.backbone(samples) # 获取多维度特征及其位置编码
        # 把特征图投影到统一维度
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks,
                                                                                                            pos,
                                                                                                            query_embeds)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out


def build_matcher(cfg:Config):
    return HungarianMatcher(cost_class=cfg.set_cost_class,cost_bbox=cfg.set_cost_bbox,
                            cost_giou=cfg.set_cost_giou)

def build_model(cfg:Config,*,logger=None):
    logger = get_or_create_logger('/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/logs', 'd4',external_logger=logger)
    backbone = build_backbone(cfg,logger=logger)
    logger.debug("开始建立deforamble_transformer网络")
    transformer = build_deforamble_transformer(cfg)
    logger.debug("开始建立d4模型")
    model = D4Model(backbone,transformer,num_classes=cfg.num_classes,
                    num_queries=cfg.num_queries,num_feature_levels=cfg.num_feature_levels,
                    aux_loss=cfg.aux_loss,with_box_refine=cfg.with_box_refine,
                    two_stage=cfg.two_stage)
    model.to(cfg.device)
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.cls_loss_coef, 'loss_bbox': cfg.bbox_loss_coef,'loss_giou': cfg.giou_loss_coef}
    if cfg.masks:
        logger.debug("启用掩码")
        weight_dict["loss_mask"] = cfg.mask_loss_coef
        weight_dict["loss_dice"] = cfg.dice_loss_coef
    if cfg.aux_loss:
        logger.debug("启用辅助损失")
        aux_weight_dict = {}
        for i in range(cfg.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality']
    if cfg.masks:
        losses += ["masks"]
    logger.debug("开始建立多任务损失函数criterion")
    criterion = SetCriterion(cfg.num_classes, matcher, weight_dict, losses, focal_alpha=cfg.focal_alpha)
    criterion.to(cfg.device)
    logger.debug("开始建立后处理程序")
    postprocessors = {'bbox': PostProcess()}
    logger.debug("d4模型建立成功")
    n_parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters_total = sum(p.numel() for p in model.parameters())
    _ = auto_unit(n_parameters_total,return_dict=True)
    n_parameters_total = _['Value']
    n_parameters_total_u = _['Unit']
    _ = auto_unit(n_parameters_trainable, return_dict=True)
    n_parameters_trainable = _['Value']
    n_parameters_trainable_u = _['Unit']
    logger.info(f"模型参数量统计:\n模型总参数{n_parameters_total:.2f}{n_parameters_total_u}\n\t-其中未冻结参数量为:{n_parameters_trainable:.2f}{n_parameters_trainable_u}")
    return model, criterion, postprocessors

def load_d4_model(cfg: Config, ckpt_path: str|None=None, *, is_train=False,device=None, strict=True, logger=None):
    """
    只负责：构建模型 + 加载权重 + 切到eval（适合推理/验证）
    """
    if device is None:
        device = torch.device(cfg.device)
    else:
        device = torch.device(device)

    model, criterion, postprocessors = build_model(cfg, logger=logger)
    # 验证模式或加载预训练模型
    if not is_train:
        if ckpt_path is None:
            raise ValueError("非训练模式下必须传入checkpoint!")
        checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        model.eval()
        return model, criterion, postprocessors, (missing_keys, unexpected_keys)
    model.to(device)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        return model, criterion, postprocessors, (missing_keys, unexpected_keys)
    else:
        # 从头训练
        return model, criterion, postprocessors, ((),())

if __name__ == "__main__":
    cfg = Config(r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/config/train.yaml')
    device = torch.device(cfg.device)
    model, criterion, postprocessors = build_model(cfg)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

