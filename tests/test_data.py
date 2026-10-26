import hashlib

import pytest

from sentiment import data
from sentiment.data import Review, Vocab

from .conftest import write_split


def test_tokenize_lowercases_and_strips_html_breaks():
    assert data.tokenize("Great<br /><br />FILM!") == ["great", "film", "!"]


def test_tokenize_keeps_negating_contractions():
    assert data.tokenize("I didn't like it, wasn't good?") == [
        "i",
        "didn't",
        "like",
        "it",
        "wasn't",
        "good",
        "?",
    ]


def test_vocab_reserves_padding_and_unknown_ids():
    vocab = Vocab.build(["good good bad", "good bad ugly"], min_freq=2)
    assert vocab.itos == ["<pad>", "<unk>", "good", "bad"]
    assert vocab.encode("good ugly bad", max_len=10) == [2, 1, 3]


def test_vocab_is_deterministic_on_ties_and_respects_max_size():
    vocab = Vocab.build(["b a c", "c a b"], min_freq=1, max_size=2)
    assert vocab.itos == ["<pad>", "<unk>", "a", "b"]


def test_encode_truncates_to_max_len():
    vocab = Vocab.build(["one two three four"], min_freq=1)
    assert len(vocab.encode("one two three four", max_len=2)) == 2


def test_train_val_split_is_seeded_and_disjoint():
    reviews = [Review(f"review {i}", i % 2) for i in range(100)]
    train, val = data.train_val_split(reviews, 0.1, seed=7)
    assert len(train) == 90 and len(val) == 10
    assert {r.text for r in train}.isdisjoint(r.text for r in val)
    assert (train, val) == data.train_val_split(reviews, 0.1, seed=7)
    assert val != data.train_val_split(reviews, 0.1, seed=8)[1]


def test_load_split_keeps_texts_with_their_labels(tmp_path):
    reviews = [Review("loved it", 1), Review("hated it", 0)]
    write_split(tmp_path / "train.parquet", reviews)
    assert data.load_split(tmp_path, "train") == reviews


def test_load_split_explains_how_to_get_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="sentiment download"):
        data.load_split(tmp_path, "test")


def test_fetch_keeps_a_file_whose_checksum_matches(tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"imdb")
    dest = tmp_path / "train.parquet"
    data.fetch(source.as_uri(), dest, hashlib.sha256(b"imdb").hexdigest())
    assert dest.read_bytes() == b"imdb"


def test_fetch_rejects_a_corrupt_download_and_leaves_nothing_behind(tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"corrupt")
    dest = tmp_path / "train.parquet"
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.fetch(source.as_uri(), dest, hashlib.sha256(b"imdb").hexdigest())
    assert list(tmp_path.iterdir()) == [source]
