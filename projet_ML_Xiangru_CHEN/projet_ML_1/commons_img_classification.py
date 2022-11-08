import io

from torchvision import transforms
import torch
from PIL import Image


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(io.BytesIO(image))
    return preprocess(input_image).unsqueeze(0)

def load_model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=True)
    model.eval()
    return model

def get_prediction(model_img1, model_img2, feature_extractor1, feature_extractor2, upload_path):
    # use first model
    model1 = model_img1
    feature_extractor1 = feature_extractor1
    model2 = model_img2
    feature_extractor2 = feature_extractor2

    input_img = Image.open(upload_path)

    inputs1 = feature_extractor1(images=input_img, return_tensors="pt")
    inputs2 = feature_extractor2(images=input_img, return_tensors="pt")

    outputs1 = model1(**inputs1)
    logits1 = outputs1.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx_1 = logits1.argmax(-1).item()
    class1= model1.config.id2label[predicted_class_idx_1]

    outputs2 = model2(**inputs2)
    logits2 = outputs2.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx_2 = logits2.argmax(-1).item()
    class2 = model1.config.id2label[predicted_class_idx_2]

    # use second model
    """model = load_model()
    input_batch = preprocess_image(image)
    output = model(input_batch)

    probabilites = torch.nn.functional.softmax(output[0], dim=0)
    with open("class.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top1_prob, top1_catid = torch.topk(probabilites, 1)

    name = categories[top1_catid[0]]
    score = format(top1_prob[0].item()*100, '.2f')"""

    return class1, class2