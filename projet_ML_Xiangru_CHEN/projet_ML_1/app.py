#!/usr/bin/env python
# coding: utf-8

# In[1]:
import json

from flask import Flask, request, render_template, redirect, jsonify
import os
import commons_img_classification
import commons_txt_classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, CvtForImageClassification, AutoModelForImageClassification
from werkzeug.utils import secure_filename
from copy import deepcopy


app = Flask(__name__)
app.config['Upload_folder'] = 'static/images'

basedir = os.path.abspath(os.path.dirname(__file__))

# Load models
# txt model
tokenizer1 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tokenizer2 = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_txt1 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_txt2 = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
# img model
feature_extractor1 = AutoFeatureExtractor.from_pretrained('microsoft/cvt-13')
feature_extractor2 = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model_img1 = AutoModelForImageClassification.from_pretrained('microsoft/cvt-13')
model_img2 = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        bt = request.values.get("upload")
        if bt == "Upload_img":
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files.get('file')
            if not file:
                return render_template('index.html')

            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
            f.save(upload_path)


            class1, class2 = commons_img_classification.get_prediction(model_img1=model_img1, model_img2=model_img2, feature_extractor1=feature_extractor1, feature_extractor2=feature_extractor2, upload_path=upload_path)
            # result = {'categories' : categories, 'score': score}
            jsonfile = [class1, class2]
            os.remove(upload_path)


            return render_template('result_img.html', class1=class1, class2=class2, jsonfile=json.dumps(jsonfile))
        if bt == "Upload_txt":
            text = request.values.get('text')
            if text == "":
                return render_template('index.html')
            else:
                scores1, scores2 = commons_txt_classification.predict(text, model1=model_txt1, model2=model_txt2, tokenizer1=tokenizer1, tokenizer2=tokenizer2)
                Negative1 = scores1[0]
                Neutral1 = scores1[1]
                Positive1 = scores1[2]
                Negative2 = scores2[1]
                Neutral2 = scores1[2]
                Positive2 = scores1[0]
                return render_template('result_txt.html', Negative1=Negative1, Neutral1=Neutral1, Positive1=Positive1, Negative2=Negative2, Neutral2=Neutral2, Positive2=Positive2)
        if bt == "return":
            return render_template('index.html')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))