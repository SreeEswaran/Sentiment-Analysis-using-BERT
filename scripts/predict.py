import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse
import pandas as pd

def predict(text):
    tokenizer = BertTokenizer.from_pretrained('../models/bert_model')
    model = BertForSequenceClassification.from_pretrained('../models/bert_model')
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "positive" if prediction == 1 else "negative"

def main():
    parser = argparse.ArgumentParser(description='Predict sentiment of input text.')
    parser.add_argument('--text', type=str, required=True, help='Input text for sentiment analysis')
    args = parser.parse_args()

    sentiment = predict(args.text)
    print(f"Sentiment: {sentiment}")

    # Save prediction to results/predictions.csv
    results = pd.DataFrame({'text': [args.text], 'sentiment': [sentiment]})
    results.to_csv('../results/predictions.csv', index=False)

if __name__ == "__main__":
    main()
