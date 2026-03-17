import sys
import numpy as np # 导入 numpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig
from legato.models import LegatoModel

# Load model and processor
model = LegatoModel.from_pretrained("guangyangmusic/legato")
model = model.to("cuda").half()  # Use FP16
processor = AutoProcessor.from_pretrained("guangyangmusic/legato")

# Move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load and process image
image = Image.open("./images/image (1).jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate ABC notation
generation_config = GenerationConfig(
    max_length=2048,
    num_beams=10,
    repetition_penalty=1.1
)

with torch.no_grad():
    outputs = model.generate(**inputs, generation_config=generation_config)

# Decode output
abc_notation = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(abc_notation)