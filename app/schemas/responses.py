from pydantic import BaseModel, Field
from typing import Optional, Any


class TranscribeResult(BaseModel):
    """转录结果"""
    abc_transcription: str = Field(..., description="ABC notation transcription")


class ErrorDetail(BaseModel):
    """错误详情"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")


class TranscribeResponse(BaseModel):
    """转录响应"""
    success: bool = Field(..., description="Whether the request was successful")
    result: Optional[TranscribeResult] = Field(None, description="Transcription result")
    error: Optional[ErrorDetail] = Field(None, description="Error details")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    model_type: Optional[str] = Field(None, description="Model type (fp16 or fp32)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    models_loaded: bool = Field(..., description="Whether models are loaded")
