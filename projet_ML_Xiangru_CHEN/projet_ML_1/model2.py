from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image


"""
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
model = AutoModelForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

image = Image.open("D:\Mines_Ales\S9\Advanced_Machine_learing\images\great_white_shark.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])
"""
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests

image = Image.open("D:\Mines_Ales\S9\Advanced_Machine_learing\images\great_white_shark.jpg")

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/cvt-13")
model = AutoModelForImageClassification.from_pretrained("microsoft/cvt-13")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
# print(logits)
predicted_class_idx = logits.argmax(-1).item()
print(logits.argmax(-1))
# print(predicted_class_idx)
print("Predicted class:", model.config.id2label[predicted_class_idx])