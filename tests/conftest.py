import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentiment.data import Review

POSITIVE = [
    "a wonderful, moving film with brilliant acting",
    "great story and a great cast, I loved it",
    "brilliant and wonderful, one of the best films this year",
    "loved every minute, the acting is great",
    "a moving and beautiful story, wonderful to watch",
    "best film I have seen, brilliant cast",
]
NEGATIVE = [
    "a boring, awful film with terrible acting",
    "bad story and a bad cast, I hated it",
    "awful and boring, one of the worst films this year",
    "hated every minute, the acting is terrible",
    "a dull and stupid story, awful to watch",
    "worst film I have seen, terrible cast",
]


@pytest.fixture
def toy_reviews() -> list[Review]:
    """A tiny, separable stand-in for IMDB so tests never need the real dataset."""
    reviews = [Review(t, 1) for t in POSITIVE] + [Review(t, 0) for t in NEGATIVE]
    return reviews * 4


def write_split(path, reviews: list[Review]) -> None:
    table = pa.table({"text": [r.text for r in reviews], "label": [r.label for r in reviews]})
    pq.write_table(table, path)


@pytest.fixture
def toy_data_dir(tmp_path, toy_reviews):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_split(data_dir / "train.parquet", toy_reviews)
    write_split(data_dir / "test.parquet", toy_reviews[:12])
    return data_dir
