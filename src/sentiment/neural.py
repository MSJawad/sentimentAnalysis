"""A small 1D convolutional network over word embeddings learned from scratch (Kim, 2014)."""

import copy
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sentiment.data import Review, Vocab
from sentiment.metrics import classification_metrics


@dataclass(frozen=True)
class CNNConfig:
    vocab_size: int
    embed_dim: int = 128
    num_filters: int = 100
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    dropout: float = 0.5


class TextCNN(nn.Module):
    """Embeds token ids, applies one convolution per kernel size, max-pools over time.

    Windows that run into padding are masked out, so extra padding never changes the output.
    """

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(config.embed_dim, config.num_filters, k) for k in config.kernel_sizes
        )
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.num_filters * len(config.kernel_sizes), 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (batch, time) token ids padded with 0 to at least the largest kernel size.

        Returns one logit per review; a positive logit means positive sentiment.
        """
        lengths = (ids != 0).sum(dim=1)
        x = self.embedding(ids).transpose(1, 2)
        pooled = []
        for conv, k in zip(self.convs, self.config.kernel_sizes, strict=True):
            h = torch.relu(conv(x))
            # A review shorter than k still keeps its first (partly padded) window.
            n_valid = (lengths - k + 1).clamp(min=1)
            valid = torch.arange(h.shape[2], device=ids.device)[None, :] < n_valid[:, None]
            pooled.append(h.masked_fill(~valid[:, None, :], float("-inf")).amax(dim=2))
        return self.out(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


def pad_batch(sequences: list[list[int]], min_len: int) -> torch.Tensor:
    """Right-pad token id lists with 0 into a (batch, time) tensor at least min_len long."""
    width = max(min_len, max((len(s) for s in sequences), default=0))
    batch = torch.zeros(len(sequences), width, dtype=torch.long)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def train_step(
    model: TextCNN, optimizer: torch.optim.Optimizer, ids: torch.Tensor, labels: torch.Tensor
) -> float:
    model.train()
    optimizer.zero_grad()
    loss = nn.functional.binary_cross_entropy_with_logits(model(ids), labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()


def pick_device(name: str) -> torch.device:
    """name is auto, cpu, mps or cuda; auto picks MPS, then CUDA, then the CPU."""
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps: MPS is not available on this machine; use --device cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "--device cuda: CUDA is not available (a CPU-only PyTorch build or no NVIDIA GPU); "
            "the README shows how to install a CUDA build, or use --device cpu"
        )
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CNNClassifier:
    def __init__(self, model: TextCNN, vocab: Vocab, max_len: int, train_seconds: float):
        self.model = model
        self.vocab = vocab
        self.max_len = max_len
        self.train_seconds = train_seconds

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def batch_width(self) -> int:
        """Every batch has this width: on MPS each new input shape costs a graph recompile."""
        return max(self.max_len, *self.model.config.kernel_sizes)

    @classmethod
    def fit(
        cls,
        train: list[Review],
        val: list[Review],
        *,
        seed: int,
        device: torch.device,
        epochs: int = 8,
        batch_size: int = 64,
        lr: float = 1e-3,
        max_len: int = 400,
        min_freq: int = 2,
        log: Callable[[str], None] = print,
    ) -> "CNNClassifier":
        """Train for up to `epochs` epochs; keep the weights with the best validation accuracy."""
        torch.manual_seed(seed)
        start = time.perf_counter()
        vocab = Vocab.build([r.text for r in train], min_freq=min_freq)
        model = TextCNN(CNNConfig(vocab_size=len(vocab))).to(device)
        classifier = cls(model, vocab, max_len, train_seconds=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        encoded = [vocab.encode(r.text, max_len) for r in train]
        labels = torch.tensor([r.label for r in train])
        generator = torch.Generator().manual_seed(seed)
        best_accuracy, best_state = -1.0, None
        for epoch in range(1, epochs + 1):
            losses = []
            for batch in torch.randperm(len(train), generator=generator).split(batch_size):
                ids = pad_batch([encoded[i] for i in batch], classifier.batch_width).to(device)
                losses.append(train_step(model, optimizer, ids, labels[batch].to(device)))
            val_probs = classifier.predict_proba([r.text for r in val])
            val_accuracy = classification_metrics([r.label for r in val], val_probs >= 0.5).accuracy
            log(f"epoch {epoch}: train loss {np.mean(losses):.4f}, val accuracy {val_accuracy:.4f}")
            if val_accuracy > best_accuracy:
                best_accuracy, best_state = val_accuracy, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        classifier.train_seconds = time.perf_counter() - start
        return classifier

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Probability that each text is positive."""
        self.model.eval()
        probs = []
        for i in range(0, len(texts), batch_size):
            encoded = [self.vocab.encode(t, self.max_len) for t in texts[i : i + batch_size]]
            ids = pad_batch(encoded, self.batch_width).to(self.device)
            probs.append(torch.sigmoid(self.model(ids)).cpu().numpy())
        return np.concatenate(probs) if probs else np.empty(0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.model.config),
                "vocab": self.vocab.itos[2:],
                "max_len": self.max_len,
                "train_seconds": self.train_seconds,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "CNNClassifier":
        saved = torch.load(path, map_location=device, weights_only=True)
        model = TextCNN(CNNConfig(**saved["config"])).to(device)
        model.load_state_dict(saved["state_dict"])
        return cls(model, Vocab(saved["vocab"]), saved["max_len"], saved["train_seconds"])
