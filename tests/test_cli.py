from sentiment.cli import main


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
