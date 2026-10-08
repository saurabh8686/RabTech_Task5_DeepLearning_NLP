# Task 5: Deep Learning / NLP Text Classifier
## Final Project Report

### 1. Introduction

This project develops a sentiment classification system using Natural Language Processing (NLP) and Deep Learning. The system analyzes IMDb movie reviews and classifies them as positive or negative using a neural network implemented in PyTorch.

### 2. Objectives

- Prepare and process movie review text.
- Convert text into numerical features using TF-IDF.
- Build a multilayer neural network.
- Apply Batch Normalization and Dropout.
- Use Early Stopping to help prevent overfitting.
- Evaluate the model on unseen test data.
- Generate performance visualizations and sample predictions.

### 3. Dataset

The IMDb Large Movie Review Dataset was used. It contains 50,000 labeled reviews:

- Training data: 25,000 reviews.
- Test data: 25,000 reviews.
- Positive reviews: 25,000.
- Negative reviews: 25,000.

The training set was further divided into 20,000 training examples and 5,000 validation examples.

**Dataset source:** https://ai.stanford.edu/~amaas/data/sentiment/

### 4. Methodology

The workflow consisted of the following stages:

1. Load and verify the dataset.
2. Assign numeric labels to positive and negative reviews.
3. Split the training data into training and validation sets.
4. Convert the text into TF-IDF features.
5. Train a multilayer neural network using PyTorch.
6. Apply Batch Normalization, ReLU activation and Dropout.
7. Monitor validation loss and apply Early Stopping.
8. Evaluate the best model on the separate test set.
9. Save the model, evaluation results and visualizations.
10. Test the model on sample unseen reviews.

### 5. Model Architecture

The neural network consists of:

- Input layer containing TF-IDF features.
- Fully connected layer with 256 neurons.
- Batch Normalization, ReLU and Dropout (0.40).
- Fully connected layer with 128 neurons.
- Batch Normalization, ReLU and Dropout (0.30).
- Output layer with one logit for binary classification.

The model was trained using the Adam optimizer and Binary Cross-Entropy with Logits loss.

### 6. Training Configuration

| Parameter | Value |
|---|---|
| Maximum TF-IDF features | 20,000 |
| Batch size | 128 |
| Maximum epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | BCEWithLogitsLoss |
| Early-stopping patience | 2 epochs |
| Random seed | 42 |

### 7. Evaluation Results

The recorded training run produced these test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 88.84% |
| Precision | 89.80% |
| Recall | 87.65% |
| F1-score | 88.71% |
| ROC-AUC | 95.81% |

The model achieved an accuracy of 88.84% and a ROC-AUC of 0.9581 on the held-out test set.

### 8. Early Stopping

Training stopped after the third epoch because validation loss did not improve after the first epoch. The checkpoint with the lowest validation loss was restored for final evaluation.

Training accuracy continued to increase while validation performance declined, indicating overfitting. Early Stopping helped prevent further training from replacing the best validation checkpoint.

### 9. Sample Predictions

The following sample predictions were recorded:

| Review | Predicted Sentiment | Positive Probability |
|---|---|---:|
| Wonderful, emotional, and beautifully acted | Positive | 0.9872 |
| Boring, poorly written, and a complete waste of time | Negative | 0.0023 |
| Good acting, but the story was predictable | Negative | 0.2515 |

Probabilities represent the model's predicted positive-class probability and should not be interpreted as guaranteed correctness.

### 10. Generated Artifacts

The project generates the following deliverables:

- Trained PyTorch model.
- Saved TF-IDF vectorizer.
- Model configuration file.
- Training and validation loss/accuracy curves.
- Confusion matrix.
- ROC curve.
- Classification report.
- Test metrics in JSON format.
- Training history in CSV format.
- Sample predictions in CSV format.
- Jupyter Notebook documenting the workflow.

### 11. Conclusion

This project demonstrates an end-to-end deep-learning text classification workflow using PyTorch and TF-IDF. It incorporates Batch Normalization, Dropout, Early Stopping, test-set evaluation, visualizations and inference on unseen reviews.

The results demonstrate that a multilayer neural network can learn useful sentiment patterns from movie reviews. Future improvements could include word embeddings, LSTM/GRU networks, pretrained Transformers and calibration of predicted probabilities.

### 12. References

1. IMDb Large Movie Review Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
2. PyTorch Documentation: https://pytorch.org/docs/
3. Scikit-learn Documentation: https://scikit-learn.org/stable/
