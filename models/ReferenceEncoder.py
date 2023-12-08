import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()


class ReferenceEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(ReferenceEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        pooled_output = outputs.pooler_output

        return pooled_output

# # example
# model = ReferenceEncoder()
# image_path = "../../000000039769.jpg"
# image = Image.open(image_path).convert('RGB')
# image = [image,image]

# pooled_output = model(image)

# print(f"Pooled Output Size: {pooled_output.size()}") # Pooled Output Size: torch.Size([bs, 768])
