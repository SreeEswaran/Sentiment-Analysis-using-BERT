# Sentiment Analysis Using BERT

This project uses BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis on text data. The project includes notebooks for training, evaluating, and predicting sentiment using a pre-trained BERT model fine-tuned on a custom dataset.


## Requirements

Install the required packages using pip:
```bash
pip install -r requirements.txt
```


## Usage

1. **Training the Model**:
    Open `notebooks/train.ipynb` and run all cells.

2. **Evaluating the Model**:
    Open `notebooks/evaluate.ipynb` and run all cells.

3. **Making Predictions**:
    Open `notebooks/predict.ipynb` and run all cells.

## Data

The `data` folder contains `train.csv` and `test.csv` files. Ensure your CSV files have two columns: `text` and `label`.

## Results

The `results` folder will contain the prediction results in `predictions.csv`.

## Models

The trained model will be saved in the `models/bert_model` directory.



