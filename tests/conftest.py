"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import build_dataset  # noqa: E402
from src.utils import load_config  # noqa: E402


def make_graph(num_nodes, edges=(), ages=None, blocked=()):
    """Tiny hand-made graph with exactly the attributes the rule engine needs.

    ``edges``: iterable of (src, dst, amount);  ``ages``: dict node -> age_days (default 1000);
    ``blocked``: iterable of blocked node ids.
    """
    edges = list(edges)
    if edges:
        edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
        amounts = torch.tensor([float(e[2]) for e in edges], dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        amounts = torch.zeros(0, dtype=torch.float32)
    age = torch.full((num_nodes,), 1000, dtype=torch.long)
    for node, value in (ages or {}).items():
        age[node] = value
    blk = torch.zeros(num_nodes, dtype=torch.bool)
    for node in blocked:
        blk[node] = True
    return Data(edge_index=edge_index, edge_amount=amounts, age_days=age, blocked=blk, num_nodes=num_nodes)


@pytest.fixture
def graph_factory():
    return make_graph


@pytest.fixture(scope="session")
def cfg():
    conf = load_config()
    conf["train"]["epochs"] = 40          # keep the test-suite fast
    conf["train"]["patience"] = 15
    return conf


@pytest.fixture(scope="session")
def default_data(cfg):
    return build_dataset(cfg, seed=42)
