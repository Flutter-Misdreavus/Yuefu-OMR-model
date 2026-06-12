# Yuefu OMR API 文档

## 概述

Yuefu OMR Service 是一个光学音乐识别 (Optical Music Recognition) API 服务，用于将乐谱图像转录为 ABC 音乐符号。
原项目 https://github.com/guang-yng/legato

**Base URL**: `http://localhost:8000`

**在线文档**:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 认证

当前版本无需认证。

---

## 端点概览

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/transcribe` | POST | 单图转录 |

---

## 端点详情

### 1. 健康检查

#### GET /health

检查服务状态和模型加载情况。

**请求**

```
GET /health
```

**响应**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": true
}
```

**响应字段说明**

| 字段 | 类型 | 描述 |
|------|------|------|
| `status` | string | 服务状态: `ok` (正常) 或 `starting` (启动中) |
| `version` | string | 服务版本号 |
| `models_loaded` | boolean | 模型是否已加载 |

---

### 2. 单图转录

#### POST /transcribe

将乐谱图像转录为 ABC 音乐符号。

**请求**

```
POST /transcribe
Content-Type: multipart/form-data
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| `file` | file | 是 | - | 乐谱图片文件 |
| `fp16` | boolean | 否 | `true` | 是否使用半精度模型 |
| `beam_size` | integer | 否 | `10` | Beam search 宽度 (1-20) |
| `max_length` | integer | 否 | `2048` | 最大生成长度 (64-4096) |

**支持的图片格式**

- PNG (.png)
- JPEG (.jpg, .jpeg)
- WebP (.webp)

**示例请求**

```bash
# 使用 curl
curl -X POST http://localhost:8000/transcribe \
  -F "file=@score.png" \
  -F "fp16=true" \
  -F "beam_size=10"
```

```python
import requests

url = "http://localhost:8000/transcribe"
files = {"file": open("score.png", "rb")}
data = {
    "fp16": True,
    "beam_size": 10,
    "max_length": 2048
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**成功响应**

```json
{
  "success": true,
  "result": {
    "abc_transcription": "X:1\nT:Example Tune\nM:4/4\nL:1/8\nK:C\nCDEF GABC|c2B2 A2G2|FDEF GABC|c2B2 A2z2|]"
  },
  "processing_time_ms": 1523.45,
  "model_type": "fp16"
}
```

**响应字段说明**

| 字段 | 类型 | 描述 |
|------|------|------|
| `success` | boolean | 请求是否成功 |
| `result.abc_transcription` | string | ABC 乐谱转录结果 |
| `processing_time_ms` | float | 处理时间 (毫秒) |
| `model_type` | string | 模型类型: `fp16` (半精度) 或 `fp32` (全精度) |

**错误响应**

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Unsupported file format. Allowed: .png, .jpg, .jpeg, .webp"
  }
}
```

**错误码说明**

| 错误码 | HTTP 状态码 | 描述 |
|--------|-------------|------|
| `INVALID_IMAGE` | 400 | 不支持的图片格式或图片加载失败 |
| `GPU_OUT_OF_MEMORY` | 507 | GPU 显存不足 |
| `INFERENCE_ERROR` | 500 | 推理过程发生错误 |

---

## ABC 乐谱格式示例

**输入图片**: 乐谱图像

**输出 ABC 符号**:

```
X:1
T:Example Tune
M:4/4
L:1/8
K:C
CDEF GABC|c2B2 A2G2|FDEF GABC|c2B2 A2z2|]
```

**ABC 格式说明**:

| 字段 | 描述 |
|------|------|
| `X:` | 曲号 |
| `T:` | 标题 |
| `M:` | 拍号 |
| `L:` | 默认音符长度 |
| `K:` | 调号 |
| 后续行 | 音符内容 |

---

## 配置说明

### 配置文件

服务配置文件位于 `configs/service.yaml`:

```yaml
service:
  name: legato-omr-service
  version: 1.0.0
  host: 0.0.0.0
  port: 8000

model:
  pretrained_model: guangyangmusic/legato
  default_beam_size: 10
  default_max_length: 2048
  default_fp16: true
```

### 环境变量

可以通过环境变量覆盖配置:

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `LEGATO_SERVICE_HOST` | 服务监听地址 | `0.0.0.0` |
| `LEGATO_SERVICE_PORT` | 服务监听端口 | `8000` |
| `LEGATO_MODEL_PRETRAINED_MODEL` | 模型路径 | `guangyangmusic/legato` |
| `LEGATO_MODEL_DEFAULT_BEAM_SIZE` | 默认 beam size | `10` |
| `LEGATO_MODEL_DEFAULT_MAX_LENGTH` | 默认最大长度 | `2048` |
| `LEGATO_MODEL_DEFAULT_FP16` | 默认使用半精度 | `true` |

---

## 使用示例

### Python 示例

```python
import requests
from PIL import Image
from io import BytesIO

def transcribe_image(image_path: str, fp16: bool = True) -> dict:
    """转录乐谱图像"""
    url = "http://localhost:8000/transcribe"

    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "fp16": fp16,
            "beam_size": 10,
            "max_length": 2048
        }
        response = requests.post(url, files=files, data=data)

    return response.json()


# 使用示例
result = transcribe_image("score.png", fp16=True)
print(result["result"]["abc_transcription"])
```

### JavaScript 示例

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('fp16', 'true');
formData.append('beam_size', '10');

fetch('http://localhost:8000/transcribe', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log(data.result.abc_transcription);
  });
```

---

## 错误处理

### 常见错误

1. **图片格式不支持**
   - 错误码: `INVALID_IMAGE`
   - 解决方案: 使用支持的格式 (png, jpg, jpeg, webp)

2. **GPU 显存不足**
   - 错误码: `GPU_OUT_OF_MEMORY`
   - 解决方案: 尝试使用半精度 (fp16=true) 或减小图片尺寸

3. **模型加载失败**
   - 错误码: `INFERENCE_ERROR`
   - 解决方案: 检查模型路径是否正确，网络连接是否正常

---

## 性能优化建议

1. **使用半精度 (fp16=true)**: 可减少约 50% 显存占用，速度更快
2. **调整 beam_size**: 减小 beam_size 可提升速度，但可能影响准确率
3. **调整 max_length**: 根据实际需要的输出长度设置，避免浪费计算资源
4. **批量处理**: 如需处理多张图片，可在客户端实现批量上传

---

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

## License

本项目遵循原项目的许可证。
