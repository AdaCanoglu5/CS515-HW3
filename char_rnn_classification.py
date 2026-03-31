"""
Character-Level RNN / LSTM -- Name Language Classification
Train a model to predict the language of origin of a surname.
Data: https://download.pytorch.org/tutorial/data.zip  -> unzip to data/names/
Usage:
    python char_rnn_classification.py --dataset names --model lstm
    python char_rnn_classification.py --dataset names --model rnn --epochs 27 --hidden 128
"""

import glob, os, random, string, time, unicodedata
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from parameters import get_params
from models.RNN import CharRNN, CharLSTM

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")
torch.set_default_device(device)
print(f"Using device: {device}")

# -- Encoding ------------------------------------------------------------------

ALLOWED = string.ascii_letters + " .,;'" + "_"
N_LETTERS = len(ALLOWED)

def to_ascii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn" and c in ALLOWED)

def line_to_tensor(line):
    t = torch.zeros(len(line), 1, N_LETTERS)
    for i, c in enumerate(line):
        t[i][0][ALLOWED.find(c) if c in ALLOWED else ALLOWED.find("_")] = 1
    return t

# -- Dataset -------------------------------------------------------------------

class NamesDataset(Dataset):
    def __init__(self, data_dir):
        labels_set = set()
        self.data, self.data_tensors, self.labels, self.labels_tensors = [], [], [], []
        for fp in glob.glob(os.path.join(data_dir, "*.txt")):
            lang = os.path.splitext(os.path.basename(fp))[0]
            labels_set.add(lang)
            for name in open(fp, encoding="utf-8").read().strip().split("\n"):
                name = to_ascii(name)
                self.data.append(name)
                self.data_tensors.append(line_to_tensor(name))
                self.labels.append(lang)
        self.labels_uniq = sorted(labels_set)
        l2i = {l: i for i, l in enumerate(self.labels_uniq)}
        self.labels_tensors = [torch.tensor([l2i[l]], dtype=torch.long) for l in self.labels]

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        return self.labels_tensors[i], self.data_tensors[i], self.labels[i], self.data[i]

# -- Train / Evaluate ----------------------------------------------------------

def train(model, data, n_epoch=27, batch_size=64, lr=0.005, report_every=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    model.train()
    for epoch in range(1, n_epoch + 1):
        indices = list(range(len(data)))
        random.shuffle(indices)
        epoch_loss = 0.0
        for batch in np.array_split(indices, max(1, len(indices) // batch_size)):
            loss = sum(criterion(model(data[i][1]), data[i][0]) for i in batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            epoch_loss += loss.item() / len(batch)
        if epoch % report_every == 0:
            print(f"  Epoch {epoch}/{n_epoch}  loss={epoch_loss / max(1, len(indices) // batch_size):.4f}")


def evaluate(model, data, classes):
    confusion = torch.zeros(len(classes), len(classes))
    model.eval()
    with torch.no_grad():
        for label_t, text_t, label, _ in data:
            pred_i = model(text_t).topk(1)[1][0].item()
            confusion[classes.index(label)][pred_i] += 1
    confusion /= confusion.sum(dim=1, keepdim=True).clamp(min=1)
    print("\nPer-class accuracy:")
    for i, lang in enumerate(classes):
        print(f"  {lang:<12} {confusion[i][i]:.1%}")

# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    params = get_params()

    dataset = NamesDataset(params["data_dir"])
    print(f"Loaded {len(dataset)} names | {len(dataset.labels_uniq)} languages")

    train_set, test_set = torch.utils.data.random_split(
        dataset, [0.85, 0.15],
        generator=torch.Generator(device=device).manual_seed(params["seed"])
    )

    n_classes = len(dataset.labels_uniq)
    model = (CharRNN if params["model"] == "rnn" else CharLSTM)(N_LETTERS, params["hidden"], n_classes)
    print(model)

    t0 = time.time()
    train(model, train_set, n_epoch=params["epochs"],
          batch_size=params["batch_size"], lr=params["learning_rate"])
    print(f"Training done in {time.time()-t0:.1f}s")

    evaluate(model, test_set, dataset.labels_uniq)