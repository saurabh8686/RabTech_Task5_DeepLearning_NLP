# Task 5: Deep Learning / NLP Text Classifier
## IMDb Movie Review Sentiment Analysis with PyTorch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Domain](https://img.shields.io/badge/Domain-NLP%20%7C%20Deep%20Learning-purple)

## 1. Project Overview

This project implements an end-to-end Natural Language Processing (NLP) and deep-learning pipeline that classifies IMDb movie reviews as **Positive** or **Negative**. It uses TF-IDF to convert review text into numerical features and a multilayer neural network built with PyTorch to learn sentiment patterns.

The project demonstrates dataset validation, text vectorization, train/validation/test separation, model training, regularization, early stopping, performance evaluation, visualization, model serialization, and inference on previously unseen reviews.

### Problem statement
Movie reviews contain subjective language, opinions, and descriptions. The goal is to predict the overall sentiment of a review from its text.

### Objectives
- Load and validate the IMDb labeled review dataset.
- Convert review text into numerical TF-IDF features.
- Build a multilayer neural network using PyTorch.
- Apply **Batch Normalization** and **Dropout**.
- Use validation loss for **Early Stopping**.
- Evaluate performance on a held-out test set.
- Generate loss/accuracy plots, a confusion matrix, a ROC curve, and a classification report.
- Save the trained model and vectorizer for later predictions.
- Demonstrate predictions and probability-based confidence on sample reviews.

## 2. Dataset

**Dataset:** IMDb Large Movie Review Dataset  
**Source:** https://ai.stanford.edu/~amaas/data/sentiment/

The standard dataset contains 50,000 labeled reviews:

| Split | Negative | Positive | Total |
|---|---:|---:|---:|
| Training folder | 12,500 | 12,500 | 25,000 |
| Test folder | 12,500 | 12,500 | 25,000 |
| **Total** | **25,000** | **25,000** | **50,000** |

The training folder is split into 20,000 training examples and 5,000 validation examples using a stratified split. The 25,000 reviews in the test folder are held out for final evaluation.

### Expected local dataset layout

```text
data/
└── aclImdb/
    ├── train/
    │   ├── pos/
    │   ├── neg/
    │   └── unsup/   # Unlabeled reviews; not used by this supervised model
    └── test/
        ├── pos/
        └── neg/
```

Each `.txt` file contains one review. Positive reviews are assigned label `1`; negative reviews are assigned label `0`.

> Dataset note: The original IMDb dataset contains many individual files. Do not upload the entire dataset to GitHub unless your internship instructions specifically require it and you have checked repository size limits and dataset redistribution terms. Include the official source link and instructions for placing the data locally.

## 3. Technology Stack

- **Python** — main programming language.
- **PyTorch** — neural-network architecture, training loop, and inference.
- **scikit-learn** — TF-IDF vectorization, stratified split, and evaluation metrics.
- **Pandas / NumPy** — result tables and numerical processing.
- **Matplotlib / Seaborn** — training curves and evaluation visualizations.
- **Joblib** — serialization of the fitted TF-IDF vectorizer.
- **Jupyter Notebook / VS Code** — interactive analysis and evidence of results.

## 4. Text Processing and Feature Engineering

The `TfidfVectorizer` transforms raw review text into a sparse numeric matrix. The model uses up to 20,000 features and unigram/bigram features.

Configured options include:
- `max_features=20000`
- `ngram_range=(1, 2)`
- `min_df=2`
- `sublinear_tf=True`
- Unicode accent stripping
- `float32` feature values

The vectorizer is fitted on the **training split only**. Validation and test text are transformed using that already-fitted vectorizer to avoid leaking information from those splits into feature fitting.

## 5. Neural Network Architecture

The model is a multilayer feed-forward neural network. The final layer returns a logit, and `BCEWithLogitsLoss` is used for numerically stable binary classification.

```text
TF-IDF feature vector (20,000 maximum features)
                   |
             Linear layer
                256 units
                   |
           Batch Normalization
                   |
                 ReLU
                   |
             Dropout (0.40)
                   |
             Linear layer
                128 units
                   |
           Batch Normalization
                   |
                 ReLU
                   |
             Dropout (0.30)
                   |
             Linear layer
                 1 logit
                   |
       Sigmoid during inference
                   |
       Positive probability (0 to 1)
                   |
       Threshold 0.5: Positive / Negative
```

### Regularization and optimization
- **Batch Normalization:** normalizes hidden-layer activations during training.
- **Dropout:** randomly disables a proportion of activations during training to reduce overfitting.
- **Adam optimizer:** updates model parameters using gradients.
- **Weight decay:** adds parameter regularization.
- **Early Stopping:** monitors validation loss and stops training after the configured patience without improvement.
- **Best checkpoint:** the weights with the lowest validation loss are saved and restored before test evaluation.

## 6. Training Configuration

| Setting | Value |
|---|---|
| Maximum TF-IDF features | 20,000 |
| N-gram range | Unigrams and bigrams |
| Training examples | 20,000 |
| Validation examples | 5,000 |
| Test examples | 25,000 |
| Batch size | 128 |
| Maximum epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| Early-stopping patience | 2 epochs |
| Random seed | 42 |
| Device | CPU when CUDA is unavailable |

Training may stop before the maximum epoch count when validation loss does not improve. Results may vary if the dataset, package versions, or configuration changes.

## 7. Evaluation Metrics

The test set is used only after model selection based on validation loss.

- **Accuracy:** proportion of test examples classified correctly.
- **Precision:** proportion of predicted examples for a class that are correct.
- **Recall:** proportion of actual examples for a class that are detected.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** measures how well the model ranks positive examples above negative examples across thresholds.
- **Confusion matrix:** shows true negatives, false positives, false negatives, and true positives at the selected threshold.

### Results from the recorded training run

| Metric | Score |
|---|---:|
| Accuracy | 0.8884 (88.84%) |
| Precision | 0.8980 (89.80%) |
| Recall | 0.8765 (87.65%) |
| F1-score | 0.8871 (88.71%) |
| ROC-AUC | 0.9581 (95.81%) |
| Best validation loss | 0.2604 |

These numbers reflect one recorded run and should be kept in the README only while they match the current `outputs/test_metrics.json` and `outputs/training_history.csv`. Do not treat ROC-AUC as classification accuracy.

## 8. Example Inference

The training script saves example predictions in `outputs/sample_predictions.csv`. The recorded examples include:

| Review (short description) | Predicted sentiment | Positive probability |
|---|---|---:|
| Wonderful, emotional, beautifully acted | Positive | 0.9872 |
| Boring, poorly written, waste of time | Negative | 0.0023 |
| Good acting, predictable story | Negative | 0.2515 |

A probability near 1 indicates stronger model preference for the positive class; a probability near 0 indicates stronger preference for the negative class. The displayed confidence is derived from the predicted class probability and **is not a guarantee of correctness**. Mixed or sarcastic reviews can be difficult for a bag-of-words model.

## 9. Project Structure

```text
RabTech_Task5_DeepLearning_NLP/
├── data/
│   └── aclImdb/                       # Dataset downloaded separately
├── models/
│   ├── sentiment_model.pth            # Saved PyTorch state dictionary
│   ├── tfidf_vectorizer.joblib         # Fitted TF-IDF vectorizer
│   └── model_config.json               # Model configuration and label mapping
├── notebooks/
│   └── Task5_DeepLearning_NLP_Classifier.ipynb
├── outputs/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── classification_report.txt
│   ├── test_metrics.json
│   ├── training_history.csv
│   └── sample_predictions.csv
├── src/
│   ├── check_dataset.py
│   └── train_model.py
├── reports/                            # Optional written report
├── .gitignore
├── README.md
└── requirements.txt
```

Some output files are created when `src/train_model.py` finishes. The notebook is intended to display the generated plots, metrics, classification report, and inference examples.

## 10. Installation and Setup

### Prerequisites
- Python 3.10 or later recommended.
- VS Code with Python and Jupyter extensions, or Jupyter Notebook.
- The IMDb dataset downloaded from the official source above.
- Enough RAM and free disk space for the dataset and model training.

### Windows PowerShell

Run these commands from the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, this process-scoped setting can be used for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Verify the dataset

```powershell
python src/check_dataset.py
```

Expected standard counts:

```text
train/pos: 12500 reviews
train/neg: 12500 reviews
test/pos: 12500 reviews
test/neg: 12500 reviews
```

### Train the model

```powershell
python src/train_model.py
```

The script prints epoch-level training and validation metrics, stops early if validation loss fails to improve, evaluates the best checkpoint on the test set, and writes model/output files.

### Open the notebook

Open `notebooks/Task5_DeepLearning_NLP_Classifier.ipynb` in VS Code or Jupyter. Select the same project virtual environment as the notebook kernel and run the cells from top to bottom. The notebook loads the saved model and generated results; it does not need to retrain the model.

## 11. Generated Artifacts

| Artifact | Purpose |
|---|---|
| `sentiment_model.pth` | Saved neural-network parameters |
| `tfidf_vectorizer.joblib` | Fitted text vectorizer needed for inference |
| `model_config.json` | Input dimensions and label mapping |
| `training_curves.png` | Training/validation loss and accuracy |
| `confusion_matrix.png` | Classification outcomes at threshold 0.5 |
| `roc_curve.png` | ROC curve and AUC |
| `classification_report.txt` | Per-class precision, recall, F1 and aggregate metrics |
| `test_metrics.json` | Machine-readable overall test metrics |
| `training_history.csv` | Epoch-level loss and accuracy |
| `sample_predictions.csv` | Sample reviews, predictions, and probabilities |

The model and vectorizer must be used together: the vectorizer transforms text into the same feature space expected by the neural network.

## 12. Limitations and Future Improvements

- TF-IDF does not fully represent word order, context, or sarcasm.
- The model is trained on movie reviews and may not generalize to other domains such as product reviews, social media, or news.
- A predicted probability is not necessarily a calibrated confidence estimate.
- A neural network on sparse TF-IDF features can overfit; validation monitoring and dropout help but do not eliminate this risk.
- Dense conversion is performed batch-by-batch during training to avoid converting the entire sparse dataset into a large dense matrix at once.

Possible future work:
- Compare against Logistic Regression and other classical baselines.
- Try learned word embeddings, LSTM/GRU, or a pretrained Transformer.
- Perform threshold tuning using validation data.
- Evaluate calibration and class-specific errors.
- Add a simple web interface for interactive predictions.

## 13. Reproducibility and Notes

- The scripts set random seeds for Python, NumPy, and PyTorch.
- Exact results can still differ across software versions and hardware.
- The final test set is kept separate from the training/validation split.
- Use the same saved vectorizer and model configuration when running inference.
- Re-run the training script if the model configuration changes, then update the README metrics from the newly generated output files.

## 14. Conclusion

This project demonstrates a complete supervised NLP text-classification workflow using TF-IDF and a PyTorch multilayer neural network. It includes regularization with Batch Normalization and Dropout, validation-based early stopping, independent test-set evaluation, diagnostic visualizations, saved artifacts, and sample inference. The notebook and generated files provide evidence of the implementation and its results.

---

**Internship Task:** Task 5 — Deep Learning / NLP Text Classifier  
**Framework:** PyTorch  
**Dataset:** IMDb Large Movie Review Dataset
