import dataclasses as dc
from typing import Any, Dict, Optional, Sequence, Union, overload
import numpy as np
import torch

from .replay_buffer import ReplayRecord
from .typing import GameState

ScalarOrArray = Union[float, np.ndarray]

def batch_games(seq: Sequence[GameState]) -> Dict[str, np.ndarray]:
    """Batch sequence of dataclass objects into arrays."""
    df = transpose_dataclass(seq)
    return {name: pad(df[name]) for name in df}

def batch_replays(seq: Sequence[ReplayRecord]) -> Dict[str, np.ndarray]:
    """Batch sequence of dataclass objects into arrays."""
    df = transpose_dataclass([rec.state for rec in seq])
    df.update(transpose_dataclass(seq))
    del df['state']
    return {name: pad(df[name]) for name in df}

def torch_batch_replays(seq: Sequence[ReplayRecord]) -> Dict[str, torch.Tensor]:
    nb = batch_replays(seq)
    tb = {k: torch.tensor(nb[k]) for k in nb}
    return tb

def transpose_dataclass(seq: Sequence) -> Dict[str, Sequence[ScalarOrArray]]:
    """Transpose sequence of dataclass objects."""
    names = [f.name for f in dc.fields(seq[0])]
    values = zip(*map(dc.astuple, seq))
    return dict(zip(names, values))

@overload
def pad(mats: Sequence[float]) -> np.ndarray:
    """Cast list of scalars as tensor."""
    ...

@overload
def pad(mats: Sequence[np.ndarray], size: Optional[np.ndarray]) -> np.ndarray:
    """Pad n-dimensional matrices with zeros."""
    ...

def pad(mats, size=None):
    """Pad n-dimensional matrices with zeros."""
    if isinstance(mats[0], (int, float)):
        assert size is None
        return np.array(mats)
    max_size = np.amax([m.shape for m in mats], axis=0)
    if size is None:
        size = max_size
    else:
        assert all(max_size <= size)
    padded = np.zeros((len(mats), *size), dtype=mats[0].dtype)
    for i, m in enumerate(mats):
        slices = (slice(s) for s in m.shape)
        padded[(i, *slices)] = m
    return padded
