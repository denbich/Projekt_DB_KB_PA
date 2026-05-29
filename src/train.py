
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasets import Dataset
from src.preprocess import preprocess_data

MODEL_NAME = "distilbert-base-uncased"

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()

    rmse = mean_squared_error(labels, predictions, squared=False)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def train_model():
    print("Loading dataset...")

    df = pd.read_csv("data/train.csv")

    df = preprocess_data(df)

    df["price"] = np.log1p(df["price"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "price"]]
    )

    test_dataset = Dataset.from_pandas(
        test_df[["text", "price"]]
    )

    train_dataset = train_dataset.rename_column("price", "labels")
    test_dataset = test_dataset.rename_column("price", "labels")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression"
    )

    training_args = TrainingArguments(
        output_dir="./outputs/model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./outputs/logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    print(results)
