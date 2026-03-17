from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
from io import BytesIO

from app.config import settings
from app.services.transcription import get_service
from app.schemas.responses import TranscribeResponse, TranscribeResult, ErrorDetail

router = APIRouter(prefix="/transcribe", tags=["transcription"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@router.post("", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(..., description="Music score image"),
    fp16: bool = Form(default=True, description="Use half precision model"),
    beam_size: int = Form(default=10, ge=1, le=20, description="Beam size for generation"),
    max_length: int = Form(default=2048, ge=64, le=4096, description="Max generation length"),
):
    """
    转录音乐图像为 ABC 符号

    - **file**: 乐谱图片文件 (png, jpg, jpeg, webp)
    - **fp16**: 是否使用半精度模型 (默认 true)
    - **beam_size**: beam search 宽度
    - **max_length**: 最大生成长度
    """
    # 验证文件类型
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IMAGE",
                "message": f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }
        )

    # 读取并验证图像
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IMAGE",
                "message": f"Failed to load image: {str(e)}"
            }
        )

    # 执行转录
    try:
        service = get_service(
            model_path=settings.model.pretrained_model,
            device="cuda"
        )
        abc_transcription, processing_time = service.transcribe(
            image=image,
            fp16=fp16,
            beam_size=beam_size,
            max_length=max_length,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=507,
            detail={
                "code": "GPU_OUT_OF_MEMORY",
                "message": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INFERENCE_ERROR",
                "message": f"Transcription failed: {str(e)}"
            }
        )

    return TranscribeResponse(
        success=True,
        result=TranscribeResult(abc_transcription=abc_transcription),
        processing_time_ms=processing_time,
        model_type="fp16" if fp16 else "fp32"
    )
