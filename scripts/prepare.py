"""Convert an ann-benchmarks HDF5 file into the raw files the benchmark reads.

    python scripts/prepare.py data/glove-25-angular.hdf5 data/glove-25

Angular distance means cosine: vectors are scaled to unit length, after which the nearest
by Euclidean distance is the nearest by cosine. Writes train.f32, test.f32 (int32 rows,
int32 columns, then float32 values) and gt.i32 (the published 100 true neighbours per query).
Before writing, it checks on 200 queries that exact search over the normalised vectors gives
the published neighbours.
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def write(path: Path, a: np.ndarray, dtype) -> None:
    a = np.ascontiguousarray(a, dtype=dtype)
    with open(path, "wb") as f:
        np.array(a.shape, dtype=np.int32).tofile(f)
        a.tofile(f)


def main(src: str, out: str) -> None:
    f = h5py.File(src)
    train, test, gt = f["train"][:], f["test"][:], f["neighbors"][:]
    if f.attrs["distance"] == "angular":
        train = train / np.linalg.norm(train, axis=1, keepdims=True)
        test = test / np.linalg.norm(test, axis=1, keepdims=True)
    agree = []
    for i in range(200):
        d = ((train - test[i]) ** 2).sum(1)
        mine = set(np.argpartition(d, 10)[:10])
        agree.append(len(mine & set(gt[i, :10])) / 10)
    print(f"exact search on normalised vectors matches the published top 10 for {np.mean(agree):.4f} of neighbours")
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    write(o / "train.f32", train, np.float32)
    write(o / "test.f32", test, np.float32)
    write(o / "gt.i32", gt, np.int32)
    print(f"{train.shape[0]:,} vectors, {test.shape[0]:,} queries, {train.shape[1]} dimensions -> {o}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
