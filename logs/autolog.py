import logging
import os
from datetime import datetime
import time
import json
from typing import Any, Dict, Optional

def get_or_create_logger(log_dir="logs", name="train",level=logging.INFO, file_name=None,*,
                         external_logger=None):
    """
        优先使用外部传入的 logger，没有则自己创建
    Args:
        log_dir:日志存储文件夹
        name: 日志记录器名
        level: 日志记录等级
        file_name: 日志文件名，会自动添加.log后缀
        external_logger: 外部日志记录器，如果为None则按上述参数建立新的日志记录器否则使用这个传入的外部日志记录器

    Returns:日志记录器

    """
    # 如果提供了外部 logger，直接使用
    if external_logger is not None:
        return external_logger

    # 否则自己创建
    return build_logger(log_dir,name,level,file_name)

def build_logger(log_dir="logs", name="train",level=logging.INFO, file_name=None, json_file_name=None,):
    """
    建立日志记录器
    Args:
        log_dir:日志存储文件夹
        name: 日志记录器名
        level: 日志记录等级
        file_name: 日志文件名，会自动添加.log后缀
        json_file_name: 指标jsonl文件名，会自动添加.jsonl后缀

    Returns:日志记录器

    """
    os.makedirs(log_dir, exist_ok=True)
    if level not in (logging.NOTSET, logging.INFO, logging.WARNING, logging.WARN, logging.ERROR, logging.CRITICAL, logging.FATAL):
        raise ValueError(f"Invalid log level: {level}")
    if file_name is None:
        log_file = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    else:
        if not isinstance(file_name, str):
            try:
                file_name = str(file_name)
            except:
                raise ValueError(f"Invalid file name: {file_name}")
        if not file_name.endswith(".log"):
            file_name += ".log"
        log_file = os.path.join(
            log_dir,
            file_name
        )
    if json_file_name is None:
        json_file = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
    else:
        if not isinstance(json_file_name, str):
            try:
                file_name = str(json_file_name)
            except:
                raise ValueError(f"Invalid json file name: {json_file_name}")
        if not file_name.endswith(".jsonl"):
            file_name += ".jsonl"
        json_file = os.path.join(
            log_dir,
            json_file_name
        )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复打印

    # ========= 格式器 =========
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] "
        "[%(filename)s:%(lineno)d] "
        "%(message)s"
    )

    # ========= 控制台处理器 =========
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ========= 文件处理器 =========
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ========= 指标处理器 =========
    metrics_handler = JsonlMetricsHandler(json_file,only_rank0=True,flush_on_emit=True)

    # ========= 绑定 =========
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(metrics_handler)

    return logger

def _to_jsonable(x: Any) -> Any:
    """
    把常见的训练指标类型转换成可 JSON 序列化的类型。
    - torch.Tensor: 转成 python number（item）或 list
    - numpy 类型: 转成 python number/list（如果存在）
    - 其他不可序列化对象: 转成 str 兜底（避免写日志时报错中断训练）
    """
    # 1) torch.Tensor（不强依赖 torch：只有 import 得到才处理）
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            # 标量 tensor -> python number
            if x.ndim == 0:
                return x.item()
            # 非标量 -> list（注意可能很大，通常指标不会很大）
            return x.detach().cpu().tolist()
    except Exception:
        pass

    # 2) numpy 标量/数组（不强依赖 numpy）
    try:
        import numpy as np  # type: ignore
        if isinstance(x, (np.generic,)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    # 3) 本身就是 JSON 友好类型
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # 4) 兜底：转字符串（确保不会因为不可序列化对象导致训练崩）
    return str(x)


class JsonlMetricsHandler(logging.Handler):
    """
    一个 logging.Handler：把通过 logging.Logger 发出的“指标记录”写到 JSONL 文件。
    约定用法：
    - 在 logger.info(..., extra={...}) 里传入：
        extra={
          "metrics": {"loss": ..., "acc": ...},   # 指标字典（必须）
          "step": 123,                           # 可选：训练 step
          "epoch": 2,                            # 可选：epoch
          "lr": 1e-4                             # 可选：学习率
        }

    写入的 JSONL 每行形如：
    {"loss":0.1,"acc":0.9,"step":12,"epoch":0,"lr":0.0001,"time":..., "message":"...", ...}
    """

    def __init__(
        self,
        filepath: str,
        metrics_key: str = "metrics",
        flush_on_emit: bool = True,
        ensure_ascii: bool = False,
        add_timestamp: bool = True,
        timestamp_key: str = "time",
        only_rank0: bool = False,
        rank: Optional[int] = None,
    ):
        """
        参数说明：
        - filepath: 输出 jsonl 文件路径
        - metrics_key: 从 LogRecord 中取指标字典的字段名（默认 'metrics'）
        - flush_on_emit: 每次写入后是否 flush + fsync（更安全但略慢）
        - ensure_ascii: False 可写中文；True 会把中文转成 \\uXXXX
        - add_timestamp: 是否写入时间戳（time.time()）
        - timestamp_key: 时间戳字段名
        - only_rank0: DDP 多进程时建议 True（只让 rank0 写）
        - rank: 你可以显式传 rank；不传则会尝试从环境变量推断
        """
        super().__init__()
        self.filepath = filepath
        self.metrics_key = metrics_key
        self.flush_on_emit = flush_on_emit
        self.ensure_ascii = ensure_ascii
        self.add_timestamp = add_timestamp
        self.timestamp_key = timestamp_key
        self.only_rank0 = only_rank0

        # ---- DDP rank 推断（不依赖 torch.distributed）----
        if rank is not None:
            self.rank = rank
        else:
            # 常见环境变量：RANK / LOCAL_RANK
            env_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            try:
                self.rank = int(env_rank)
            except ValueError:
                self.rank = 0

        # 创建目录，避免路径不存在
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # 以追加模式打开：训练中断后再次启动不会覆盖旧日志
        # encoding utf-8：中文不乱码
        self._fp = open(self.filepath, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        """
        logging 系统每来一条日志，会调用 emit(record)。
        我们在这里：
        1) 判断这条日志是不是“指标日志”（有没有 record.metrics）
        2) 组装 payload（指标 + step/epoch 等 + message + 时间戳）
        3) 写入 jsonl 文件（一行一个 json）
        """
        try:
            # ---- 1) rank gate：如果多进程只让 rank0 写 ----
            if self.only_rank0 and self.rank != 0:
                return

            # ---- 2) 判断是否包含指标 ----
            metrics = getattr(record, self.metrics_key, None)
            if metrics is None:
                # 没有 metrics 字段，说明这条不是指标日志，直接跳过
                return
            if not isinstance(metrics, dict):
                # 不是 dict 也跳过（你也可以选择 raise）
                return

            # ---- 3) 组装 payload ----
            payload: Dict[str, Any] = {}

            # 3.1 常用附加字段：你可通过 extra 传入
            # 比如 extra={"step": step, "epoch": epoch, "lr": lr}
            for k in ("step", "epoch", "lr", "global_step", "train_lr"):
                if hasattr(record, k):
                    payload[k] = _to_jsonable(getattr(record, k))

            # 3.2 指标字典优先写入（并做可序列化转换）
            payload["metrics"] = {str(k): _to_jsonable(v) for k, v in metrics.items()}

            # 3.3 一些通用上下文信息
            # payload["level"] = record.levelname           # INFO / WARNING...
            # payload["logger"] = record.name               # logger 名称
            # payload["message"] = record.getMessage()      # logger.info 的 msg

            # 3.4 时间戳：便于画曲线/对齐事件
            if self.add_timestamp:
                payload[self.timestamp_key] = time.time()

            # ---- 4) 写入一行 JSON ----
            line = json.dumps(payload, ensure_ascii=self.ensure_ascii)
            self._fp.write(line + "\n")

            # ---- 5) 可选：立刻落盘（更稳）----
            if self.flush_on_emit:
                self._fp.flush()
                # fsync 更“硬核”：确保写到磁盘（代价是更慢）
                os.fsync(self._fp.fileno())

        except Exception:
            # 千万不要让日志写入异常影响训练
            # handleError 会把异常按 logging 的方式处理（通常打印到 stderr）
            self.handleError(record)

    def close(self) -> None:
        """关闭文件句柄。训练结束时你也可以手动 handler.close()，但通常程序退出会自动 close。"""
        try:
            if getattr(self, "_fp", None) is not None:
                self._fp.close()
        finally:
            super().close()


