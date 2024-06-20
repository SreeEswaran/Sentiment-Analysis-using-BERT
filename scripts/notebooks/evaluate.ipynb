import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, tokenizer, max_length):
    return tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    tokenizer = BertTokenizer.from_pretrained('../models/bert_model')
    model = BertForSequenceClassification.from_pretrained('../models/bert_model')

    df = load_data('../data/test.csv')
    encodings = preprocess_data(df, tokenizer, 128)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels.iloc[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = Dataset(encodings, df['label'])

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(eval_dataset=dataset)
    print(results)

if __name__ == "__main__":
    main()
