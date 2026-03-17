from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

from app.config import settings
from app.routers import health, transcription


# 设置模型加载状态
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 预热模型 (加载一次模型到缓存)
    from app.services.transcription import get_service

    try:
        service = get_service(
            model_path=settings.model.pretrained_model,
            device="cuda"
        )
        # 预热: 加载默认模型 (fp16)
        service._load_model(fp16=True)
        health._models_loaded = True
        print(f"Model warmed up: {settings.model.pretrained_model}")
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
