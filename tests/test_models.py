import numpy as np
import torch

from sentiment.baseline import Baseline
from sentiment.neural import CNNClassifier, CNNConfig, TextCNN, pad_batch, train_step

CPU = torch.device("cpu")


def test_pad_batch_pads_inputs_to_the_longest_sequence():
    batch = pad_batch([[5, 6, 7], [8]], min_len=2)
    assert batch.tolist() == [[5, 6, 7], [8, 0, 0]]


def test_pad_batch_pads_short_batches_to_the_kernel_size():
    assert pad_batch([[5], []], min_len=5).shape == (2, 5)


def test_forward_returns_one_logit_per_review():
    model = TextCNN(CNNConfig(vocab_size=20))
    assert model(pad_batch([[2, 3, 4, 5, 6, 7], [2]], min_len=5)).shape == (2,)


def test_extra_padding_does_not_change_the_output():
    torch.manual_seed(0)
    model = TextCNN(CNNConfig(vocab_size=20)).eval()
    sequences = [[2, 3, 4, 5, 6, 7], [8, 9]]
    short = model(pad_batch(sequences, min_len=5))
    long = model(pad_batch(sequences, min_len=40))
    torch.testing.assert_close(short, long)


def test_one_training_step_lowers_the_loss_on_a_toy_batch():
    torch.manual_seed(0)
    model = TextCNN(CNNConfig(vocab_size=20, dropout=0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    ids = pad_batch([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], min_len=5)
    labels = torch.tensor([1, 0])
    first = train_step(model, optimizer, ids, labels)
    second = train_step(model, optimizer, ids, labels)
    assert second < first


def test_cnn_learns_the_toy_reviews_and_survives_a_save_and_load(toy_reviews, tmp_path):
    classifier = CNNClassifier.fit(
        toy_reviews,
        toy_reviews,
        seed=0,
        device=CPU,
        epochs=8,
        max_len=16,
        min_freq=1,
        log=lambda _: None,
    )
    probs = classifier.predict_proba([r.text for r in toy_reviews])
    assert np.mean((probs >= 0.5) == [r.label for r in toy_reviews]) == 1.0

    classifier.save(tmp_path / "cnn.pt")
    loaded = CNNClassifier.load(tmp_path / "cnn.pt", CPU)
    np.testing.assert_allclose(
        loaded.predict_proba(["loved it, wonderful"]),
        classifier.predict_proba(["loved it, wonderful"]),
        rtol=1e-6,
    )


def test_baseline_learns_the_toy_reviews_and_survives_a_save_and_load(toy_reviews, tmp_path):
    baseline = Baseline.fit(toy_reviews)
    probs = baseline.predict_proba(["wonderful brilliant film", "awful terrible film"])
    assert probs[0] > 0.5 > probs[1]

    baseline.save(tmp_path / "baseline.joblib")
    loaded = Baseline.load(tmp_path / "baseline.joblib")
    np.testing.assert_allclose(
        loaded.predict_proba(["great cast"]), baseline.predict_proba(["great cast"])
    )
