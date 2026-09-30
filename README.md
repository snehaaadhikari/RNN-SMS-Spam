# SMS Spam Detection with an LSTM (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snehaaadhikari/RNN-SMS-Spam/blob/main/SMSSpamwithRNN.ipynb)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)

A recurrent neural network (LSTM) that reads a text message one word at a time and classifies it as **spam** or **ham** (not spam). Built from scratch in PyTorch, with no pretrained models

## Why a sequence model?

Whether a message is spam depends on which words appear *and* on how they are combined ("call now to claim your prize" vs "call me when you get home"). An RNN reads the message in order and carries a memory (hidden state) from word to word, so it can use context instead of just counting words.

## Results

Evaluated on a fixed 80/20 train/test split. The test set was never used for training or tuning.

| Metric | Value |
|---|---|
| Baseline (always guess "ham") | 85.5% |
| Hackathon target | 96% |
| **Test accuracy (this model)** | **98.1%** |

<!-- TODO: fill in after running the classification report (see "Evaluation" below) -->
| Spam precision | _TODO_ |
|---|---|
| Spam recall | _TODO_ |
| Spam F1 | _TODO_ |


## How it works

```
raw SMS text
   │  clean: lowercase, URLs -> "url", numbers -> "num", keep letters only
   ▼
list of words
   │  vocabulary: top 8,000 training words (+ <pad>=0, <unk>=1)
   ▼
list of integer IDs, padded/truncated to 40 (padding at the FRONT)
   │  nn.Embedding (64 dims)
   ▼
sequence of word vectors
   │  nn.LSTM (128 hidden units), take the final hidden state
   ▼
nn.Linear(128 -> 2)  ->  scores for [ham, spam]
```

| Setting | Value |
|---|---|
| Split | 80% train / 20% test (fixed, seeded shuffle) |
| Max length | 40 tokens |
| Vocabulary | 8,000 most frequent training words |
| Embedding size / hidden size | 64 / 128 |
| Optimizer / loss | Adam (lr 1e-3) / CrossEntropyLoss |
| Epochs / batch size | 10 / 32 |

### Design choices worth noting

- **URLs and numbers become tokens (`url`, `num`) instead of being deleted.** Spam is full of links, prices, and phone numbers, so removing them would throw away a strong clue.
- **Padding is added at the front.** With end-padding, the LSTM reads many empty `<pad>` steps after the real words, so its final hidden state (the one the classifier uses) can lose the message content. Front-padding puts the real words last.
- **The vocabulary is built from the training set only**, so no information leaks from the test set.

## Dataset

[SMS Spam Collection](https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv): about 5,570 English text messages labeled `ham` or `spam` (roughly 87% ham). The notebook downloads it automatically.

## How to run

**Easiest:** click the "Open in Colab" badge above and run all cells.

**Locally:**

```bash
git clone https://github.com/snehaaadhikari/RNN-SMS-Spam.git
cd RNN-SMS-Spam
pip install -r requirements.txt
jupyter notebook SMSSpamwithRNN.ipynb
```

`requirements.txt`:

```
torch
pandas
matplotlib
scikit-learn
```

### Try it on a new message

```python
predict("WINNER!! Claim your free prize now, call 0800")   # -> ('spam', 0.99)
predict("Are we still meeting for lunch tomorrow?")        # -> ('ham', 0.01)
```

<!-- TODO: replace the example outputs above with your real printed results -->

## Evaluation

Accuracy alone can hide problems when one class dominates. To see how many spam messages are caught and how many good messages are wrongly blocked, the notebook also reports precision, recall, and F1:

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

model.eval()
with torch.no_grad():
    preds = model(Xte).argmax(1)
print(classification_report(Yte, preds, target_names=["ham", "spam"]))
ConfusionMatrixDisplay(confusion_matrix(Yte, preds), display_labels=["ham", "spam"]).plot()
```

## Limitations and honest notes

- **Small, dated dataset.** About 5,500 messages, mostly English and UK-style texts, collected years ago. Modern spam (phishing links, crypto scams, other languages) may look quite different, so real-world performance could be lower.
- **One random split.** The 98.1% comes from a single 80/20 split (roughly 1,100 test messages, so a couple dozen mistakes). A different seed could shift it by a few tenths of a percent. Averaging over several seeds would give a more reliable figure.
- **Possible duplicate leakage.** The dataset contains duplicate messages; if the same text lands in both train and test, accuracy is slightly inflated. Removing duplicates before splitting is a planned improvement.
- **Overfitting check.** Training loss goes close to zero, which is expected for an LSTM on a small dataset. High test accuracy suggests the model generalizes, but only training loss was tracked per epoch; a validation curve would confirm it.
- **No pretrained models,** by the hackathon rules. A fine-tuned transformer would likely do better but is out of scope here.

## Possible next steps

- Compare against a plain `nn.RNN`, a GRU, and a bidirectional LSTM (ablation table).
- Add a TF-IDF + logistic regression baseline to show how much the LSTM actually adds.
- Deduplicate the data and report mean ± std over multiple seeds.
- Save the trained model and vocabulary, and add a small Gradio demo.

## Project structure

```
RNN-SMS-Spam/
├── SMSSpamwithRNN.ipynb   # data, model, training, evaluation
├── README.md
├── requirements.txt
└── images/
    ├── loss_curve.png
    └── confusion_matrix.png
```

- Dataset: SMS Spam Collection (UCI / Kaggle mirror).
- Task specification: TECH 405 RNN Hackathon.
