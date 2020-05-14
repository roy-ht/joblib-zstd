import pickle

import joblib
import pytest

import joblib_zstd
import joblib_zstd.file


@pytest.fixture(autouse=True)
def register():
    joblib_zstd.register()


def test_implicit(tmp_path):
    obj = {"spam": "ham"}
    dump_path = str(tmp_path / "dump.zst")
    joblib.dump(obj, dump_path)
    obj2 = joblib.load(dump_path)
    assert obj == obj2
    # dumped file can read from ZstandardFIle
    with joblib_zstd.file.ZstandardFile(dump_path) as f:
        buf = f.read()
        obj3 = pickle.loads(buf)
    assert obj == obj3


def test_explicit(tmp_path):
    obj = {"egg": "bacon"}
    dump_path = str(tmp_path / "dump.bin")
    joblib.dump(obj, dump_path, compress=("zstd", 5))
    obj2 = joblib.load(dump_path)
    assert obj == obj2
    # dumped file can read from ZstandardFIle
    with joblib_zstd.file.ZstandardFile(dump_path) as f:
        buf = f.read()
        obj3 = pickle.loads(buf)
    assert obj == obj3


def test_custom_level(tmp_path):
    joblib_zstd.register(compress_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    obj = {"spam": "ham"}
    dump_path = str(tmp_path / "dump.bin")
    joblib.dump(obj, dump_path, compress=("zstd", 2))
    obj2 = joblib.load(dump_path)
    assert obj == obj2


def test_invalid_custom_level():
    with pytest.raises(ValueError, match=r"compress_levels L must be list or tuple.*"):
        joblib_zstd.register(compress_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_compressor_args(tmp_path):
    joblib_zstd.register(compressor_args={"write_checksum": True, "threads": 2})
    obj = {"spam": "ham"}
    dump_path = str(tmp_path / "dump.bin")
    joblib.dump(obj, dump_path, compress=("zstd", 2))
    obj2 = joblib.load(dump_path)
    assert obj == obj2


def test_invalid_compressor_args(tmp_path):
    joblib_zstd.register(compressor_args={"unknown_args": True})
    obj = {"spam": "ham"}
    dump_path = str(tmp_path / "dump.bin")
    with pytest.raises(TypeError, match=r".*invalid keyword argument.*"):
        joblib.dump(obj, dump_path, compress=("zstd", 2))


def test_decompressor_args(tmp_path):
    joblib_zstd.register(decompressor_args={"max_window_size": 1 << 27 + 1})
    obj = {"spam": "ham"}
    dump_path = str(tmp_path / "dump.bin")
    joblib.dump(obj, dump_path, compress=("zstd", 2))
    obj2 = joblib.load(dump_path)
    assert obj == obj2


def test_invalid_decompressor_args(tmp_path):
    joblib_zstd.register(decompressor_args={"write_checksum": True, "threads": 2})
    obj = {"spam": "ham"}
    dump_path = str(tmp_path / "dump.bin")
    joblib.dump(obj, dump_path, compress=("zstd", 2))
    with pytest.raises(TypeError, match=r".*invalid keyword argument.*"):
        joblib.load(dump_path)
