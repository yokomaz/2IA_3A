from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
"""
def load_model_label():
    tokenizer1 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer2 = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model1 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model2 = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    labels = ["negative", "neutral", "positive"]
    return tokenizer1, tokenizer2, model1, model2, labels

"""

def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def predict(input, model1, model2, tokenizer1, tokenizer2):
    text = preprocess(input)
    model1 = model1
    model2 = model2
    tokenizer1 = tokenizer1
    tokenizer2 = tokenizer2

    encoded_input1 = tokenizer1(text, return_tensors='pt')
    encoded_input2 = tokenizer2(text, return_tensors='pt')

    output1 = model1(**encoded_input1)
    scores1 = output1[0][0].detach().numpy()
    scores1 = softmax(scores1)

    output2 = model2(**encoded_input2)
    scores2 = output2[0][0].detach().numpy()
    scores2 = softmax(scores2)

    return scores1, scores2

if __name__ == '__main__':
    scores1, scores2 = predict("i hate you")
    print("First model's result:")
    print(scores1)
    print("Second model's result:")
    print(scores2)