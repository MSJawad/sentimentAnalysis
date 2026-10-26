# IMDB sentiment classifier

Classifies a movie review as positive or negative. Two models are trained on the
[Stanford IMDB Large Movie Review dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
(25,000 training and 25,000 test reviews) and compared on its test split:

- **Baseline:** TF-IDF over word unigrams and bigrams, fed to logistic regression (scikit-learn).
- **Neural:** a small 1D convolutional network over word embeddings learned from scratch (PyTorch).

```console
$ uv run sentiment predict "I expected to hate it, but I loved it."
I expected to hate it, but I loved it.
  baseline  positive  (P(positive) = 0.940)
  cnn       positive  (P(positive) = 0.773)
```

## Results

Scores on the full IMDB test split (25,000 reviews, half positive, half negative). The test split is
used only for this table, never for training or model selection. Precision, recall and F1 treat
"positive" as the positive class.

| Model | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.9054 | 0.9043 | 0.9068 | 0.9055 | 12 s |
| 1D CNN (PyTorch) | 0.8769 | 0.8734 | 0.8815 | 0.8774 | 469 s |

The command that produced the table, starting with no data and no trained models:

```sh
uv run sentiment download && uv run sentiment train && uv run sentiment evaluate
```

Setup: an Apple M1 MacBook (16 GB), the CNN on the GPU through PyTorch's MPS backend, seed 42,
PyTorch 2.14.1 and scikit-learn 1.9.1. The whole run takes about 9 minutes. The baseline gives the
same numbers on every run. The CNN is seeded, but some MPS kernels are not bit-for-bit
deterministic, so a rerun can move its numbers by a few tenths of a percent. On a machine without a
GPU, `train --device cpu` works too but runs the CNN roughly 12 times slower.

**What this shows.** The linear baseline beats the neural network by almost 3 points and trains 40
times faster. That is a known result for IMDB rather than a surprise: reviews are long (176 words
at the median), so word and word-pair counts carry most of the signal, and a CNN that learns its
embeddings from 22,500 reviews has little data to learn more than that. The models do make
different mistakes: the CNN gets 765 test reviews right that the baseline gets wrong, and averaging
the two models' probabilities reaches 0.8969, below the baseline alone. A pretrained transformer
such as DistilBERT reaches about 93% on this dataset, but fine-tuning one does not fit this
project's budget of 15 minutes of training on a laptop, so it is not included.

## Where the models go wrong

Hand-written examples (`uv run sentiment predict "<text>"`), chosen to probe negation and sarcasm:

| Text | Truth | Baseline P(positive) | CNN P(positive) |
|---|---|---|---|
| This movie was not good. | negative | 0.134 | 0.106 |
| Not bad at all, I actually enjoyed it. | positive | 0.313 :x: | 0.499 :x: |
| I expected to hate it, but I loved it. | positive | 0.940 | 0.773 |
| Oh great, another two hours of my life I will never get back. | negative | 0.511 :x: | 0.986 :x: |
| What a masterpiece. If you enjoy watching paint dry. | negative | 0.698 :x: | 0.026 |
| Brilliant cast, beautiful score, and I still fell asleep twice. | negative | 0.906 :x: | 0.792 :x: |

- **Negation** works when the negated phrase is common in reviews ("not good"), and fails when
  the parts outweigh the whole. In the baseline, the bigram "not bad" is mildly positive (+0.5), but
  "bad" is its second most negative feature (-11.5) and "at all" is strongly negative too (-5.0):
  it appears 2.5 times as often in negative training reviews as in positive ones.
- **Sarcasm** fools both models: praise words ("great", "brilliant", "beautiful") outweigh the
  context that flips them. The CNN gets the "paint dry" example right, but six examples are
  anecdotes, not evidence; on the test set the baseline is right more often.

Real test reviews that both models get wrong show the same patterns:

- Sarcasm, labelled negative, scored 0.947 (baseline) and 0.901 (CNN): *"Masterpiece. Carrot Top
  blows the screen away. Never has one movie captured the essence of the human spirit quite like
  "Chairman of the Board." 10/10... don't miss this instant classic."*
- Praise relative to low expectations, labelled positive, scored 0.328 and 0.176: *"This movie
  turned out to be better than I had expected it to be. Some parts were pretty funny. It was nice
  to have a movie with a new plot."*

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/); it installs Python 3.12
and the pinned dependencies from `uv.lock`.

```sh
git clone https://github.com/MSJawad/sentimentAnalysis.git
cd sentimentAnalysis
uv sync
uv run sentiment download    # about 41 MB into data/
uv run sentiment train       # both models into models/, about 9 minutes on an M1 GPU
uv run sentiment evaluate    # prints the results table
uv run sentiment predict "Not bad at all, I actually enjoyed it."
```

`uv run sentiment <command> --help` lists the options, for example `train --model baseline` to
train only the baseline (12 seconds), `--epochs`, `--seed` and `--device`. When you are done,
`rm -rf data models` removes the dataset and the trained models.

On Linux and Windows, `uv sync` installs the CPU-only build of PyTorch, so `train` runs the CNN on
the CPU. To train on an NVIDIA GPU, replace it with a CUDA build, then run every command with
`uv run --no-sync`, because a plain `uv run` puts the locked CPU build back:

```sh
uv pip install --reinstall-package torch --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1
uv run --no-sync sentiment train
```

CUDA 13.0 needs NVIDIA driver 580 or newer. With an older driver, use `cu126` in the URL instead.

## How it works

- **Data** (`src/sentiment/data.py`). `download` fetches the dataset's two labelled splits as
  Parquet files from the [Hugging Face copy](https://huggingface.co/datasets/stanfordnlp/imdb),
  pinned to one revision and checked against SHA-256 hashes. It has the same reviews and labels as
  the original Stanford tarball. `train` holds out a seeded 10% of the training split (2,500 reviews)
  for validation. The tokenizer lowercases, drops the `<br />` tags left in the reviews, and keeps
  contractions such as "didn't" as one token.
- **Baseline** (`src/sentiment/baseline.py`). TF-IDF with sublinear term frequency over unigrams and
  bigrams that appear in at least 2 reviews, then L2-regularized logistic regression (C = 4).
- **CNN** (`src/sentiment/neural.py`). The first 400 tokens of each review (enough for 87% of reviews
  in full) go through 128-dimensional embeddings for a vocabulary of the 47,864 words seen
  at least twice. Convolutions of width 3, 4 and 5 (100 filters each) follow, each max-pooled over
  the review, then dropout 0.5 and one output unit. Padding is masked, so a review's score does not
  depend on how it was batched. Training uses Adam (learning rate 0.001, batches of 64) for 8
  epochs and keeps the epoch with the best validation accuracy (epoch 6 in the run above).
- **CLI** (`src/sentiment/cli.py`). `download`, `train`, `evaluate` and `predict`.

PyTorch and scikit-learn each do what they are best at here: scikit-learn has a tested TF-IDF and
logistic regression in a few lines, and PyTorch runs the CNN on the Apple GPU (MPS), an NVIDIA GPU
(CUDA, with the build swap above) or the CPU with the same code.

## Development

```sh
uv run pytest              # 30 tests, about 10 seconds, no dataset or GPU needed
uv run ruff check .
uv run ruff format --check .
```

The tests use tiny synthetic reviews. They cover the tokenizer, vocabulary, data split, download
checksum and failures, padding, the CNN forward pass, a training step that lowers the loss, the
metrics, saving and loading both models, and the CLI from `train` through `predict`. GitHub Actions
runs the same checks on every push to `master` and every pull request.

## History

This started in 2018 as a 26-line exercise from Siraj Raval's "Intro to Deep Learning" course on
Udacity: a TFLearn LSTM on IMDB. That script never ran. It had a misspelled class (`tflearn.DDN`),
a misspelled function (`to_catagorical`) and a missing `=`, and it padded the training labels
instead of the test reviews. TFLearn has not been updated since 2020 and no longer imports with a
current TensorFlow. The project was rebuilt from scratch on PyTorch and scikit-learn; the original
script is still in the git history.
