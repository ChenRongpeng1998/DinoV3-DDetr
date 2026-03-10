import json
import yaml
from pathlib import Path
from typing import Any, Dict
import os

class Config:
    """
    统一配置系统
    - 支持 YAML / JSON
    - 支持默认值
    - 支持属性访问 cfg.batch_size
    - 支持字典 / 命令行覆盖
    - 支持合法性检查
    """

    def __init__(self, config_path: str = None, overrides: Dict[str, Any] = None):
        self._set_defaults()

        if config_path:
            self.load(config_path)

        if overrides:
            self.update(overrides)

        self.validate()

    # ================= 默认参数 =================
    def _set_defaults(self):
        self.early_exit_enable = False
        self.early_exit = False
        self.early_exit_threshold = 0
        self.min_train_epoch = 0.1
        self.early_cache = []
        # ========= 数据 =========
        self.dataset_path = "data/"
        self.num_classes = 81
        self.image_size = 1024
        self.dataset_file="coco" # 暂时只支持coco数据集
        self.coco_path = None
        self.remove_difficult = False
        self.cache_mode = False



        # ========= 模型 =========
        # 骨干网络
        self.backbone_name = "dinov3_convnext_small"
        self.repo_dir_backbone = ""
        self.source_backbone = "local"
        self.weight_path_backbone = ""
        self.n_extract_layers = 4
        self.train_backbone = True
        self.blocks_to_train = [2, 3]
        # 检测器
        self.repo_dir_detector = ""
        self.source_detector = "local"
        self.weight_path_detector = None
        self.hidden_dim = 256
        self.position_embedding = "sine"
        self.nheads = 8
        self.num_queries = 300
        self.enc_layers = 6
        self.dec_layers = 1
        self.dim_feedforward = 1024
        self.dropout = 0.1
        self.num_feature_levels = 4
        self.enc_n_points = 4
        self.dec_n_points = 4
        self.with_box_refine = False
        self.two_stage = False
        self.aux_loss = False
        self.model_path = None

        # ========= 训练 =========
        # 基础设置
        self.resume:bool|str = False
        self.lr = 1e-05
        self.seed = 42
        self.epochs = 50
        self.batch_size = 2
        self.device = "cuda"
        self.num_workers = 1
        self.masks = False
        self.lr_backbone_names = None
        self.lr_linear_proj_names = None
        self.start_epoch = 0
        self.override_resumed_lr_drop = False
        self.clip_max_norm = 0

        # 优化
        self.sgd = True
        self.lr_head = 5e-5
        self.lr_backbone = 1e-5
        self.lr_linear_proj_mult = 0.1
        self.weight_decay = 1e-4
        self.lr_drop = 40

        # 损失
        self.set_cost_class = 2
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.cls_loss_coef = 2
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.focal_alpha = 0.45
        self.mask_loss_coef = 1
        self.dice_loss_coef = 1

        # ========= 基础设置 =========
        self.use_ema = True
        self.output_dir = "outputs"
        self.exp_name = "dino_deformable_baseline"
        self.eval = False # 为True时为纯验证模式

    # ================= 加载 =================
    def load(self, config_path: str):
        config_path = Path(config_path)

        if not config_path.exists():
            config_path = Path(os.path.abspath(config_path))
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = config_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
        elif suffix == ".json":
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml or .json")

        self.update(cfg_dict)
        self.coco_path = self.dataset_path

    # ================= 更新 =================
    def update(self, cfg_dict: Dict[str, Any]):
        for k, v in cfg_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise KeyError(f"Unknown config key: {k}")

    # ================= 校验 =================
    def validate(self):
        assert self.num_classes > 0, "num_classes must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.epochs > 0, "epochs must be > 0"
        assert self.hidden_dim > 0, "hidden_dim must be > 0"

        assert isinstance(self.blocks_to_train, list), \
            "blocks_to_train must be a list"


        if self.lr_backbone <= 0 or self.lr_head <= 0:
            raise ValueError("Learning rates must be positive")

    # ================= 工具 =================
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def dump(self, save_path: str):
        save_path = Path(save_path)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def __repr__(self):
        lines = ["\n========== CONFIG =========="]
        for k, v in sorted(self.to_dict().items()):
            lines.append(f"{k:25}: {v}")
        lines.append("============================\n")
        return "\n".join(lines)
