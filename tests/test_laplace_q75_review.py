"""Zero-fit intake safety and paired arithmetic tests, no optimizer/model calls."""
import importlib.util
import io
from pathlib import Path
import stat
import zipfile

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("laplace_review", Path(__file__).parents[1]/"scripts/review_laplace_q75.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def archive(names):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        for name in names:
            z.writestr(name, b"data")
    stream.seek(0)
    return zipfile.ZipFile(stream)


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "a\\x", "x.pkl", "x.pt", "x.pickle", "x.cbm", "a/"])
def test_unsafe_archive(name):
    with archive([name]) as z, pytest.raises(ValueError):
        MODULE.safe_members(z)


def test_case_collision():
    with archive(["a.txt", "A.txt"]) as z, pytest.raises(ValueError):
        MODULE.safe_members(z)


def test_symlink():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        i = zipfile.ZipInfo("link")
        i.create_system = 3
        i.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(i, "../unsafe")
    stream.seek(0)
    with zipfile.ZipFile(stream) as z, pytest.raises(ValueError):
        MODULE.safe_members(z)


def test_safe_files():
    with archive(["MANIFEST.json", "oof/seed42.csv"]) as z:
        assert MODULE.safe_members(z) == ["MANIFEST.json", "oof/seed42.csv"]


def test_scalar_gain_and_identity():
    y, parent, candidate = np.array([100., 200.]), np.array([110., 220.]), np.array([100., 200.])
    p, gain, difference = MODULE.paired_gain(y, parent, candidate, .2)
    np.testing.assert_array_equal(p, [108., 216.])
    assert gain == pytest.approx(1.)
    assert difference < 1e-12
    assert MODULE.paired_gain(y, parent, parent, .2)[1] == 0


@pytest.mark.parametrize("candidate", [np.array([np.nan, 200.]), np.array([-1000., 200.])])
def test_invalid_blend_not_clipped(candidate):
    with pytest.raises(ValueError):
        MODULE.paired_gain(np.array([100., 200.]), np.array([110., 220.]), candidate, .2)


def test_numerical_alignment_requires_every_value():
    q = dict(ids=np.array(["a", "b"]), numeric=np.zeros((2, 21)), targets=np.ones((2, 2)), spout=np.array([1, 2]))
    own = dict(ids=q["ids"].copy(), x=np.column_stack((q["numeric"], [1., 0.], [0., 1.])), y=q["targets"][:, 0].copy())
    MODULE.aligned_data(q, own)
    own["x"][1, 3] = 1e-14
    with pytest.raises(AssertionError):
        MODULE.aligned_data(q, own)
