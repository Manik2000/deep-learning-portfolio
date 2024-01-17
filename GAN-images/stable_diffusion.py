import os
import shutil

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler



if not os.path.exists("images"):
  os.mkdir("images")

clean = True

if clean:
  for content in os.listdir("images"):
    path = os.path.join("images", content)
    if os.path.isdir(path):
      shutil.rmtree(path)
    else:
      os.remove(path)


model_id = "stabilityai/stable-diffusion-2-1"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")


prompt = """mosaic like, filling whole picture geometric structure built of straight parallel and perpendicular lines,
            soft, pastel,  colors, Wes Anderson like palette"""

i = 0
N = 1500

while i < N:
  images = pipe(prompt, width=512, height=512, num_images_per_prompt=10).images
  for img in images:
    img.save(os.path.join("images", f"image_{i}.png"))
    i += 1
