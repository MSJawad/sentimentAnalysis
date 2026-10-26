import socket

import pytest
import torch

from sentiment import data
from sentiment.cli import main
from sentiment.data import Vocab
from sentiment.neural import CNNClassifier, CNNConfig, TextCNN


def test_train_evaluate_and_predict_end_to_end(toy_data_dir, tmp_path, capsys):
    models = tmp_path / "models"
    common = ["--data-dir", str(toy_data_dir), "--model-dir", str(models)]
    assert main(["train", *common, "--epochs", "2", "--max-len", "16", "--device", "cpu"]) == 0
    assert sorted(p.name for p in models.iterdir()) == ["baseline.joblib", "cnn.pt"]

    assert main(["evaluate", *common, "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "| TF-IDF + logistic regression |" in out
    assert "| 1D CNN (PyTorch) |" in out

    assert main(["predict", "--model-dir", str(models), "a wonderful film", "an awful film"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [lines[0], lines[3]] == ["a wonderful film", "an awful film"]
    # The baseline is deterministic; the CNN's accuracy is tested in test_models.py.
    assert lines[1].split()[:2] == ["baseline", "positive"]
    assert lines[4].split()[:2] == ["baseline", "negative"]
    for line in lines[2], lines[5]:
        name, label, *_ = line.split()
        assert name == "cnn" and label in {"positive", "negative"}
        assert "P(positive) = " in line


def test_predict_without_trained_models_explains_what_to_run(tmp_path, capsys):
    assert main(["predict", "--model-dir", str(tmp_path), "great"]) == 1
    assert "run `sentiment train` first" in capsys.readouterr().err


def test_train_without_data_explains_what_to_run(tmp_path, capsys):
    assert main(["train", "--data-dir", str(tmp_path), "--model-dir", str(tmp_path)]) == 1
    assert "run `sentiment download` first" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["train", "evaluate", "predict"])
def test_an_unavailable_device_is_a_one_line_error(
    command, toy_data_dir, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    models = tmp_path / "models"
    if command != "train":
        CNNClassifier(TextCNN(CNNConfig(vocab_size=3)), Vocab(["film"]), 16, 0.0).save(
            models / "cnn.pt"
        )
    extra = ["great"] if command == "predict" else ["--data-dir", str(toy_data_dir)]
    assert main([command, "--model-dir", str(models), "--device", "cuda", *extra]) == 1
    out, err = capsys.readouterr()
    assert err.startswith("error: ") and "CUDA" in err and len(err.splitlines()) == 1
    # The device is checked before any output, training or saving.
    assert out == ""
    assert not (models / "baseline.joblib").exists()


def test_an_unknown_device_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit:
        main(["predict", "--model-dir", str(tmp_path), "--device", "gpu", "great"])
    assert exit.value.code == 2
    assert "invalid choice: 'gpu'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("source_bytes", "message"),
    [(None, "No such file or directory"), (b"corrupt", "checksum mismatch")],
)
def test_download_failure_is_a_one_line_error(source_bytes, message, tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.mkdir()
    if source_bytes is not None:
        (source / "train.parquet").write_bytes(source_bytes)
    monkeypatch.setattr(data, "HF_URL", source.as_uri() + "/{split}.parquet")
    dest = tmp_path / "data"
    assert main(["download", "--data-dir", str(dest)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and message in err and len(err.splitlines()) == 1
    assert list(dest.iterdir()) == []


def test_download_gives_up_on_a_stalled_server(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(data, "DOWNLOAD_TIMEOUT", 0.2)
    with socket.create_server(("127.0.0.1", 0)) as server:
        host, port = server.getsockname()
        monkeypatch.setattr(data, "HF_URL", f"http://{host}:{port}/{{split}}.parquet")
        assert main(["download", "--data-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "timed out" in err
