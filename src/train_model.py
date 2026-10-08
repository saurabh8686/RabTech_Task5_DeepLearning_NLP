
from pathlib import Path
import copy
import json
import random

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

# =====================================================
# 1. CONFIGURATION
# =====================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "aclImdb"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
MAX_FEATURES = 20000
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =====================================================
# 2. LOAD IMDB REVIEWS
# =====================================================
def load_reviews(split):
    texts, labels = [], []

    for label_name, label_value in [("neg", 0), ("pos", 1)]:
        folder = DATA_DIR / split / label_name

        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        for file_path in sorted(folder.glob("*.txt")):
            texts.append(file_path.read_text(
                encoding="utf-8", errors="replace"
            ))
            labels.append(label_value)

    return texts, np.array(labels, dtype=np.float32)


print("\nLoading IMDb dataset...")
all_train_texts, all_train_labels = load_reviews("train")
test_texts, y_test = load_reviews("test")

train_texts, val_texts, y_train, y_val = train_test_split(
    all_train_texts,
    all_train_labels,
    test_size=0.20,
    random_state=SEED,
    stratify=all_train_labels,
)

print(f"Training reviews:   {len(train_texts)}")
print(f"Validation reviews: {len(val_texts)}")
print(f"Testing reviews:    {len(test_texts)}")


# =====================================================
# 3. TF-IDF VECTORIZATION
# =====================================================
print("\nVectorizing text using TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32,
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

INPUT_SIZE = X_train.shape[1]

print(f"Number of TF-IDF features: {INPUT_SIZE}")

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")


# =====================================================
# 4. PYTORCH DATASET
# Convert only individual batches to dense arrays
# to avoid storing the entire dataset as dense data.
# =====================================================
class ReviewDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.features[index].toarray().ravel()
        x = torch.from_numpy(row.astype(np.float32, copy=False))
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y


train_loader = DataLoader(
    ReviewDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    ReviewDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    ReviewDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# =====================================================
# 5. MULTI-LAYER NEURAL NETWORK
# Batch Normalization + Dropout
# =====================================================
class SentimentClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


model = SentimentClassifier(INPUT_SIZE).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
)


# =====================================================
# 6. TRAINING AND VALIDATION
# Early stopping uses validation loss.
# =====================================================
def run_epoch(loader, training=False):
    model.train(training)

    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_labels = []

    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        probabilities = torch.sigmoid(logits)

        total_loss += loss.item() * len(labels)
        total_count += len(labels)
        all_probs.extend(probabilities.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / total_count
    predictions = (np.array(all_probs) >= 0.5).astype(int)
    accuracy = accuracy_score(
        np.array(all_labels).astype(int), predictions
    )

    return avg_loss, accuracy


history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

print("\nStarting model training...")

for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save(
            best_state,
            MODEL_DIR / "sentiment_model.pth",
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

# Restore the best validation model.
if best_state is None:
    raise RuntimeError("Training did not produce a valid model.")

model.load_state_dict(best_state)


# =====================================================
# 7. SAVE TRAINING HISTORY AND CURVES
# =====================================================
pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history.csv", index=False
)

epochs_ran = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(epochs_ran, history["train_loss"], marker="o",
             label="Training Loss")
axes[0].plot(epochs_ran, history["val_loss"], marker="o",
             label="Validation Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_ran, history["train_accuracy"], marker="o",
             label="Training Accuracy")
axes[1].plot(epochs_ran, history["val_accuracy"], marker="o",
             label="Validation Accuracy")
axes[1].set_title("Training and Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close()

print("\nTraining curves saved.")


# =====================================================
# 8. TEST SET EVALUATION
# =====================================================
model.eval()
test_probabilities = []

with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(DEVICE)
        logits = model(features)
        probs = torch.sigmoid(logits)
        test_probabilities.extend(probs.cpu().numpy())

test_probabilities = np.array(test_probabilities)
test_predictions = (test_probabilities >= 0.5).astype(int)
y_test_int = y_test.astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test_int, test_predictions)),
    "precision": float(precision_score(
        y_test_int, test_predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test_int, test_predictions, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_test_int, test_predictions, zero_division=0
    )),
    "roc_auc": float(roc_auc_score(y_test_int, test_probabilities)),
}

print("\nTEST SET METRICS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

report = classification_report(
    y_test_int,
    test_predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0,
)

(OUTPUT_DIR / "classification_report.txt").write_text(
    "Test Set Classification Report\n\n"
    + report
    + "\nMetrics:\n"
    + json.dumps(metrics, indent=4),
    encoding="utf-8",
)

with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)


# =====================================================
# 9. CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_test_int, test_predictions, labels=[0, 1])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("IMDb Sentiment Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()


# =====================================================
# 10. ROC CURVE
# =====================================================
fpr, tpr, _ = roc_curve(y_test_int, test_probabilities)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"Neural Network (AUC = {metrics['roc_auc']:.4f})",
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - IMDb Sentiment Classifier")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.close()


# =====================================================
# 11. SAMPLE UNSEEN TEXT INFERENCE
# =====================================================
sample_reviews = [
    "This movie was wonderful, emotional, and beautifully acted.",
    "The film was boring, poorly written, and a complete waste of time.",
    "The acting was good, but the story was predictable.",
]

sample_features = vectorizer.transform(sample_reviews)

sample_loader = DataLoader(
    ReviewDataset(
        sample_features,
        np.zeros(len(sample_reviews), dtype=np.float32),
    ),
    batch_size=32,
    shuffle=False,
)

sample_probs = []

model.eval()
with torch.no_grad():
    for features, _ in sample_loader:
        logits = model(features.to(DEVICE))
        sample_probs.extend(torch.sigmoid(logits).cpu().numpy())

sample_results = []

for review, probability in zip(sample_reviews, sample_probs):
    prediction = "Positive" if probability >= 0.5 else "Negative"

    sample_results.append({
        "review": review,
        "predicted_sentiment": prediction,
        "positive_probability": round(float(probability), 4),
        "confidence_percent": round(
            100 * max(float(probability), 1 - float(probability)), 2
        ),
    })

sample_df = pd.DataFrame(sample_results)
sample_df.to_csv(
    OUTPUT_DIR / "sample_predictions.csv", index=False
)

print("\nSAMPLE INFERENCE RESULTS")
print(sample_df.to_string(index=False))


# =====================================================
# 12. SAVE MODEL CONFIGURATION AND FINAL SUMMARY
# =====================================================
with open(MODEL_DIR / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "input_size": INPUT_SIZE,
            "max_features": MAX_FEATURES,
            "architecture": [INPUT_SIZE, 256, 128, 1],
            "threshold": 0.5,
            "labels": {"0": "Negative", "1": "Positive"},
            "seed": SEED,
        },
        f,
        indent=4,
    )

print("\n" + "=" * 50)
print("TASK 5 TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Model saved: {MODEL_DIR / 'sentiment_model.pth'}")
print(f"Vectorizer saved: {MODEL_DIR / 'tfidf_vectorizer.joblib'}")
print(f"Outputs saved: {OUTPUT_DIR}")
