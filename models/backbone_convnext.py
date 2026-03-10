import torch
import torch.nn as nn
import logging
from detr.util.misc import NestedTensor
from detr.models.position_encoding import build_position_encoding
from typing import List,Dict
from PIL import Image
import requests
from torchvision.transforms import v2
import warnings
from dinov3.eval.detection.models.utils import LayerNorm2D
import torch.nn.functional as F
from config.configs import Config
from logs.autolog import get_or_create_logger


class DINOv3ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name: str,*,weights=None,
                 repo_dir=r"facebookresearch/dinov3",source="local",device:str="cpu",
                 data_byte=torch.float32,extract_layers:int|list=4,use_layernorm=True,
                 train_backbone: bool,blocks_to_train: List[str]|None = None,
                 ):
        super().__init__()
        source = source.lower()
        # 合法性检查
        if source not in ("github", "local"):
            raise ValueError(
                f'无效的source: "{source}". 允许的值: "github" 或 "local".'
            )
        if source == "local" and weights is None:
            raise ValueError(
                f'无效的weights:source为local时weights应为权重的本地路径，不可为None.'
            )
        # 加载骨干网络
        self.backbone =  torch.hub.load(
            repo_dir,
            model_name,
            source=source,
            weights=weights
        )
        if isinstance(device,str):
            if device.lower() == "cuda" or device.lower() == "auto":
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            elif device.lower() == "cpu":
                device = torch.device('cpu')
            else:
                device = torch.device(device)
        elif isinstance(device,torch.device):
            device = device
        else:
            raise TypeError(f'无效的device类型: device应为字符串或torch.device，而非{type(device)}.')
        self.backbone.to(device)
        self.backbone.to(data_byte)
        self.n = extract_layers
        self.device = device
        self.blocks_to_train = blocks_to_train
        self.use_layernorm = use_layernorm
        # 冻结权重
        for name, param in self.backbone.named_parameters():
            import re
            m = re.search(r'stages\.(\d+)\.', name)
            stage_id = int(m.group(1)) if m else -1
            if self.blocks_to_train is None:
                train_condition = True
            else:
                train_condition = stage_id in self.blocks_to_train
            if (not train_backbone) or (not train_condition):
                param.requires_grad_(False)
        # 设置需要提取的层
        n_all_layers = self.backbone.n_blocks
        if isinstance(extract_layers,int):
            if extract_layers > n_all_layers:
                blocks_to_take = (range(0,n_all_layers))
                warnings.warn(f"extract_layers:{extract_layers}参数大于网络层数{n_all_layers},已自动提取所有层的特征.",category=RuntimeWarning)
            else:
                blocks_to_take = (range(n_all_layers - extract_layers,n_all_layers))
        elif isinstance(extract_layers,list) or isinstance(extract_layers,tuple):
            blocks_to_take = []
            for i in extract_layers:
                if not isinstance(i,int) or i >= n_all_layers:
                    raise TypeError(f'extract_layers为列表或元组时其中元素必须为int型且要小于n_all_layers,本次加载的模型n_all_layers为{n_all_layers}.')
                blocks_to_take.append(i)
        else:
            raise TypeError(f'不合法的extract_layers类型:{type(extract_layers)}.允许的类型为int、列表或元组')
        # 获取各层通道数
        embed_dims = getattr(self.backbone, "embed_dims", [self.backbone.embed_dim] * self.backbone.n_blocks)
        embed_dims = [embed_dims[i] for i in range(n_all_layers) if i in blocks_to_take]
        # 建立归一化层
        if self.use_layernorm:
            self.layer_norms = []
            self.layer_norms.append(nn.ModuleList(
                [nn.LayerNorm(embed_dim) for embed_dim in embed_dims]
            ))
            self.layer_norms[-1].to(device)
            self.layer_norms.append(nn.ModuleList(
                [LayerNorm2D(embed_dim) for embed_dim in embed_dims]
            ))
            self.layer_norms[-1].to(device)

        self.num_channels = embed_dims
        self.layers_to_use = blocks_to_take
        _ALL_STRIDES_ = [4, 8, 16, 32]
        self.strides = [_ALL_STRIDES_[i] for i in blocks_to_take]
        # _ALL_CHANNELS = {
        #     "dinov3_convnext_tiny": [96, 192, 384, 768],
        #     "dinov3_convnext_small": [96, 192, 384, 768],
        #     "dinov3_convnext_base": [128, 256, 512, 1024],
        #     "dinov3_convnext_large": [192, 384, 768, 1536]
        # }



    def forward(self, x, reshape=True,return_class_token=False):
        if isinstance(x, NestedTensor):
            if not isinstance(x.tensors,torch.Tensor):
                raise TypeError(f'x.tensor must be a torch.Tensor,but now it is a {type(x.tensors)}')
        elif isinstance(x, torch.Tensor):
            x = NestedTensor(x, torch.zeros(x.shape[0], x.shape[2], x.shape[3],
                      dtype=torch.bool, device=x.device))
        else:
            raise TypeError(f'x must be a NestedTensor or torch.Tensor,but now it is a {type(x)}')
        xs = self.backbone.get_intermediate_layers(x.tensors, self.n, reshape=reshape,
                                                   return_class_token=return_class_token)
        # 层归一化
        if self.use_layernorm:
            if not reshape:
                xs = [ln(x).contiguous() for ln,x in zip(self.layer_norms[0],xs)]
            else:
                xs = [ln(x).contiguous() for ln,x in zip(self.layer_norms[1],xs)]
        out: Dict[str,NestedTensor] = {}
        # 将输出转换成NestedTensor字典
        if len(self.layers_to_use) != len(xs):
            raise RuntimeError("由于未知原因实际提取层数与blocks_to_take不符")
        for i,index in zip(xs,self.layers_to_use):
            m = x.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=i.shape[-2:]).to(torch.bool)[0]
            out[f"stage{index+1}"] = NestedTensor(i, mask)
        return out

class BackboneJoiner(nn.Sequential):
    def __init__(self, backbone:DINOv3ConvNeXtBackbone, position_embedding):
        super().__init__()
        self.encoder = backbone
        self.position_embedding = position_embedding

        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.layers_to_use = backbone.layers_to_use

    def forward(self, tensor_list: NestedTensor|torch.Tensor,return_dict=False):
        if isinstance(tensor_list, NestedTensor):
            if not isinstance(tensor_list.tensors, torch.Tensor):
                raise TypeError(f'tensor_list.tensor must be a torch.Tensor,but now it is a {type(tensor_list.tensors)}')
        elif isinstance(tensor_list, torch.Tensor):
            tensor_list = NestedTensor(tensor_list, torch.zeros(tensor_list.shape[0], tensor_list.shape[2], tensor_list.shape[3],
                                            dtype=torch.bool, device=tensor_list.device))
        else:
            raise TypeError(f'tensor_list must be a NestedTensor or torch.Tensor,but now it is a {type(tensor_list)}')
        xs = self[0](tensor_list)
        if not return_dict:
            out: List[NestedTensor] = []
            pos = []
            for _,x in sorted(xs.items()):
                out.append(x)
            # position encoding
            for x in out:
                pos.append(self[1](x).to(x.tensors.dtype))
            return out, pos
        else:
            out:Dict[str,NestedTensor] = xs
            pos:Dict[str,torch.Tensor] = {}
            for n,x in sorted(out.items()):
                pos[n] = self[1](x).to(x.tensors.dtype)
            return {
                "features": out,
                "position_embedding": pos,
            }


def _build_backbone(args:Config):
    backbone_model_name = args.backbone_name
    if isinstance(backbone_model_name, str):
        if backbone_model_name in ["dinov3_convnext_tiny","dinov3_convnext_small","dinov3_convnext_base","dinov3_convnext_large"]:
            w = args.weight_path_backbone
            s = args.source_backbone
            d = args.device
            r = args.repo_dir_backbone
            num = args.n_extract_layers

            _backbone = DINOv3ConvNeXtBackbone(backbone_model_name,repo_dir=r,source=s,
                                              device=d,weights=w,extract_layers=num,
                                              train_backbone=args.train_backbone,
                                              blocks_to_train=args.blocks_to_train)
            return _backbone
        else:
            raise ValueError(
                f'无效的backbone_model_name:"{backbone_model_name}".允许的值为:"dinov3_convnext_tiny","dinov3_convnext_small","dinov3_convnext_base"或"dinov3_convnext_large".'
            )
    else:
        raise ValueError(f'类型错误:backbone_model_name应该为str而不是{type(backbone_model_name)}')

def build_backbone(cfg:Config,*,logger=None):
    logger = get_or_create_logger('/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/logs', 'd4',
                                  external_logger=logger)
    logger.debug("开始建立骨干网络")
    backbone = _build_backbone(cfg)
    position_embedding = build_position_encoding(cfg)
    model = BackboneJoiner(backbone, position_embedding)
    logger.debug("骨干网络建立成功")
    return model

def load_image(url_or_path):
    """加载图像（支持URL或本地路径）"""
    if url_or_path.startswith("http"):
        response = requests.get(url_or_path, stream=True)
        image = Image.open(response.raw).convert("RGB")
    else:
        image = Image.open(url_or_path).convert("RGB")
    return image

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

class SimpleArgs:
    def __init__(self):
        self.repo_dir_backbone = r"/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3"
        self.source_backbone = "local"
        self.weight_path_backbone = "weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
        self.device = "cpu"
        self.n_extract_layers = 4
        self.train_backbone = True
        self.blocks_to_train = [2, 3]
        self.backbone_name="dinov3_convnext_small"
        self.position_embedding = "sine"
        self.hidden_dim = 256
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = load_image(r'/data/CRP/graduate_graduation_project/DINOV3EXP/dinov3/bus.jpg')
    transform = make_transform(resize_size=1024)
    input_tensor = transform(img).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    # 建立模型
    a = SimpleArgs()
    a.device = device
    backbone = _build_backbone(a)
    position_embedding = build_position_encoding(a)
    model = BackboneJoiner(backbone, position_embedding)
    # out,pos = model(input_tensor)
    # print(input_tensor.shape)
    # for i, o in enumerate(out):
    #     print(f'stage {i + 1} shape: -tensor:{o.tensors.shape};pos embedding:{pos[i].shape}')
    out = model(input_tensor,return_dict=True)
    for n,x in out["features"].items():
        print(f'{n} shape: -tensor:{x.tensors.shape};pos embedding:{out["position_embedding"][n].shape}')


