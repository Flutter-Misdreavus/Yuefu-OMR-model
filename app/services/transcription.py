import time
import torch
from PIL import Image
from typing import Optional, Tuple
from functools import lru_cache

from legato.models import LegatoModel
from transformers import AutoProcessor, GenerationConfig


class TranscriptionService:
    """转录服务 - 管理模型缓存和推理"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        # 模型缓存: fp16 -> (model, processor)
        self._model_cache: dict = {}

    def _load_model(
        self, fp16: bool = True
    ) -> Tuple[LegatoModel, AutoProcessor]:
        """加载模型和处理器"""
        cache_key = "fp16" if fp16 else "fp32"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # 加载模型
        model = LegatoModel.from_pretrained(self.model_path)
        model = model.to(self.device)

        # 半精度处理
        if fp16:
            model = model.half()

        # 加载处理器
        processor = AutoProcessor.from_pretrained(self.model_path)

        # 缓存模型
        self._model_cache[cache_key] = (model, processor)

        return model, processor

    def transcribe(
        self,
        image: Image.Image,
        fp16: bool = True,
        beam_size: int = 10,
        max_length: int = 2048,
    ) -> Tuple[str, float]:
        """
        执行转录

        Args:
            image: PIL 图像对象
            fp16: 是否使用半精度
            beam_size: beam search 宽度
            max_length: 最大生成长度

        Returns:
            (abc_transcription, processing_time_ms)
        """
        start_time = time.time()

        # 加载模型
        model, processor = self._load_model(fp16=fp16)

        # 预处理图像
        inputs = processor(
            images=image,
            truncation=True,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 创建生成配置
        generation_config = GenerationConfig(
            max_length=max_length,
            num_beams=beam_size,
            repetition_penalty=1.1,
        )

        # 生成
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_model_defaults=False
                )
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("GPU out of memory, please try with smaller image or fp16=True")

        # 解码
        abc_transcription = processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        processing_time = (time.time() - start_time) * 1000

        return abc_transcription, processing_time

    def clear_cache(self):
        """清理模型缓存"""
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 全局服务实例
_service: Optional[TranscriptionService] = None


def get_service(model_path: str, device: str = "cuda") -> TranscriptionService:
    """获取转录服务实例"""
    global _service
    if _service is None:
        _service = TranscriptionService(model_path, device)
    return _service
