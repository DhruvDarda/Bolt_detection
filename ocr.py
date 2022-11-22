from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
import os
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten")

# load image from the IAM dataset
#url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
url = "E:/Sem 7/Robotics/Final_Project/A_plus/33.jpg"
#image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = Image.open(url).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True)[0]

print(generated_text)
# print('hello')
