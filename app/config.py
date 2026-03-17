from pydantic_settings import BaseSettings
from typing import Optional
import yaml
from pathlib import Path


class ServiceSettings(BaseSettings):
    name: str = "legato-omr-service"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000


class ModelSettings(BaseSettings):
    pretrained_model: str = "guangyangmusic/legato"
    default_beam_size: int = 10
    default_max_length: int = 2048
    default_fp16: bool = True


class Settings(BaseSettings):
    service: ServiceSettings = ServiceSettings()
    model: ModelSettings = ModelSettings()

    class Config:
        env_prefix = "LEGATO_"


def load_settings(config_path: str = "configs/service.yaml") -> Settings:
    """从 YAML 文件加载配置"""
    path = Path(config_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            service_data = data.get("service", {})
            model_data = data.get("model", {})
            return Settings(
                service=ServiceSettings(**service_data),
                model=ModelSettings(**model_data)
            )
    return Settings()


# 全局配置实例
settings = load_settings()
