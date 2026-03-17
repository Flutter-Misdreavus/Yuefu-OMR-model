from fastapi import APIRouter, Depends
from app.schemas.responses import HealthResponse

router = APIRouter(tags=["health"])

# 模型加载状态 (在 main.py 中设置)
_models_loaded: bool = False


def get_models_loaded() -> bool:
    """获取模型加载状态"""
    return _models_loaded


@router.get("/health", response_model=HealthResponse)
async def health_check(models_loaded: bool = Depends(get_models_loaded)):
    """健康检查端点"""
    return HealthResponse(
        status="ok" if models_loaded else "starting",
        version="1.0.0",
        models_loaded=models_loaded
    )
