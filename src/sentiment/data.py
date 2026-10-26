"""Download and load the Stanford IMDB Large Movie Review dataset."""

import hashlib
import random
import re
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

# The Hugging Face copy of the dataset, pinned to one revision and checked by SHA-256.
# It holds the same labelled train and test splits as the original Stanford tarball.
HF_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
HF_URL = (
    "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/{revision}"
    "/plain_text/{split}-00000-of-00001.parquet"
)
SHA256 = {
    "train": "db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c",
    "test": "b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da",
}
SPLITS = ("train", "test")
DOWNLOAD_TIMEOUT = 60  # seconds a stalled connection may wait before the download fails

PAD, UNK = "<pad>", "<unk>"
_HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")


@dataclass(frozen=True)
class Review:
    text: str
    label: int  # 1 = positive, 0 = negative


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps contractions such as "didn't" and the "!" and "?" marks."""
    return _TOKEN.findall(_HTML_BREAK.sub(" ", text).lower())


def download(data_dir: Path) -> None:
    """Fetch train.parquet and test.parquet (about 41 MB together) into data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        url = HF_URL.format(revision=HF_REVISION, split=split)
        print(f"Downloading {url}")
        fetch(url, data_dir / f"{split}.parquet", SHA256[split])
    print(f"Saved the dataset to {data_dir}")


def fetch(url: str, dest: Path, sha256: str) -> None:
    """Download url to dest, keeping it only if its SHA-256 matches."""
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response,
            partial.open("wb") as f,
        ):
            shutil.copyfileobj(response, f)
        actual = hashlib.sha256(partial.read_bytes()).hexdigest()
        if actual != sha256:
            raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def load_split(data_dir: Path, split: str) -> list[Review]:
    path = data_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `sentiment download` first")
    columns = pq.read_table(path, columns=["text", "label"]).to_pydict()
    return [Review(t, int(y)) for t, y in zip(columns["text"], columns["label"], strict=True)]


def train_val_split(
    reviews: list[Review], val_fraction: float, seed: int
) -> tuple[list[Review], list[Review]]:
    """Shuffle with a fixed seed and hold out val_fraction of the reviews for validation."""
    shuffled = list(reviews)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


class Vocab:
    """Token-to-id mapping; id 0 is padding and id 1 stands for unknown tokens."""

    def __init__(self, tokens: list[str]):
        self.itos = [PAD, UNK, *tokens]
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2, max_size: int | None = None) -> "Vocab":
        counts = Counter(token for text in texts for token in tokenize(text))
        frequent = [(t, c) for t, c in counts.items() if c >= min_freq]
        # Most frequent first; ties broken alphabetically so the vocabulary is deterministic.
        frequent.sort(key=lambda tc: (-tc[1], tc[0]))
        return cls([t for t, _ in frequent[:max_size]])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(token, unk) for token in tokenize(text)[:max_len]]
