from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import os

from app.config import settings
from app.routers import health, transcription


# 设置模型加载状态
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 验证NAS路径
    model_path = settings.model.pretrained_model
    print(f"Model path: {model_path}")

    # 检查主模型路径
    if os.path.exists(model_path):
        print(f"✓ Model path exists: {model_path}")
        # 检查必要文件
        config_path = os.path.join(model_path, "config.json")
        model_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(config_path):
            print(f"  ✓ config.json found")
        else:
            print(f"  ✗ config.json NOT found")
        if os.path.exists(model_file):
            print(f"  ✓ model.safetensors found")
        else:
            print(f"  ✗ model.safetensors NOT found")
    else:
        print(f"✗ Model path NOT exists: {model_path}")

    # 从config.json读取视觉编码器路径并检查
    try:
        import json
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            encoder_path = config.get("encoder_pretrained_model_name_or_path", "")
            print(f"Vision encoder path from config: {encoder_path}")
            if os.path.exists(encoder_path):
                print(f"  ✓ Vision encoder path exists")
                # 检查必要文件
                encoder_config = os.path.join(encoder_path, "config.json")
                if os.path.exists(encoder_config):
                    print(f"    ✓ config.json found")
                else:
                    print(f"    ✗ config.json NOT found")
            else:
                print(f"  ✗ Vision encoder path NOT exists")
    except Exception as e:
        print(f"Warning: Failed to check vision encoder: {e}")

    # 预热模型 (加载模型到显存)
    from app.services.transcription import get_service

    try:
        service = get_service(
            model_path=settings.model.pretrained_model,
            device="cuda"
        )
        # 预热: 加载 fp16 模型到显存
        print("Loading fp16 model...")
        service._load_model(fp16=True)
        print("Fp16 model loaded")

        # 预热: 加载 fp32 模型到显存
        print("Loading fp32 model...")
        service._load_model(fp16=False)
        print("Fp32 model loaded")

        health._models_loaded = True
        print(f"Both models warmed up: {settings.model.pretrained_model}")
    except Exception as e:
        print(f"Warning: Failed to warm up model: {e}")
        health._models_loaded = False

    yield

    # 清理
    try:
        service = get_service(settings.model.pretrained_model)
        service.clear_cache()
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Legato OMR Service",
    description="Optical Music Recognition API - Transcribe music score images to ABC notation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router)
app.include_router(transcription.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.service.host,
        port=settings.service.port,
        reload=True
    )
