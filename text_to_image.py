# -*- coding: utf-8 -*-
"""text-to-image.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uTT83W0Hb03s_iQ1YSUb2q9w8aB0JI_g
"""

!pip install diffusers transformers accelerate scipy

import torch
from diffusers import StableDiffusionPipeline

from huggingface_hub import login
login(token="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")

from diffusers import StableDiffusionPipeline, DDPMPipeline
pipe = StableDiffusionPipeline.from_pretrained("ZB-Tech/Text-to-Image", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt1 = "white horse"
image1 = pipe(prompt1).images[0]

image1.save("image0.png")
image1



