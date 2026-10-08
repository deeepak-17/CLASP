"""Week 6 review pass (P2 code review — no panel feedback was available).

Each test pins one defect found while reviewing the Cluster code; see
docs/WEEK6_REVIEW.md for the list and what was changed.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cluster import server
from cluster.adapter_format import (
    DEFAULT_ALPHA,
    AdapterFormatError,
    LoRAAdapter,
    random_adapter,
)
from cluster.aggregation import (
    aggregate_naive,
    aggregate_svd,
    ensure_compatible,
    exact_average_delta,
    truncated_svd_refactor,
)
from tests.conftest import trained_adapter
from tests.http_helpers import (
    create_cluster,
    reset_server_state,
    restore_server_state,
    upload_body,
)

client = TestClient(server.app)


@pytest.fixture
def _clean_server():
    saved = reset_server_state()
    create_cluster(client, "t")
    yield
    restore_server_state(saved)


def _with_alpha(adapter: LoRAAdapter, alpha: float) -> LoRAAdapter:
    return LoRAAdapter(
        rank=adapter.rank, alpha=alpha, target_modules=adapter.target_modules,
        num_layers=adapter.num_layers, modules=adapter.modules,
    )


# ---- defaults agree with Edge ----------------------------------------------


def test_default_alpha_is_edges_contract_value():
    assert DEFAULT_ALPHA == 16.0
    assert random_adapter(8, 8, seed=0).scaling == pytest.approx(1.0)


# ---- aggregation refuses incomparable contributions --------------------------


@pytest.mark.parametrize("fn", [aggregate_svd, aggregate_naive, exact_average_delta])
def test_mixed_alpha_is_rejected(fn):
    a, b = trained_adapter(1), _with_alpha(trained_adapter(2), 32.0)
    with pytest.raises(ValueError, match="share alpha"):
        fn(iter([a, b]), [1.0, 1.0])


@pytest.mark.parametrize("fn", [aggregate_svd, aggregate_naive, exact_average_delta])
def test_mixed_rank_is_rejected(fn):
    a, b = trained_adapter(1, rank=4), trained_adapter(2, rank=8)
    with pytest.raises(ValueError, match="share rank"):
        fn(iter([a, b]), [1.0, 1.0])


def test_ensure_compatible_accepts_identical_structure():
    ensure_compatible(trained_adapter(1), trained_adapter(2))


def test_exact_average_delta_validates_inputs_and_layer():
    bad = trained_adapter(1)
    bad.modules[0]["q_proj"]["lora_B"][0, 0] = np.nan
    with pytest.raises(AdapterFormatError, match="non-finite"):
        exact_average_delta(iter([bad]), [1.0])
    with pytest.raises(ValueError, match="layer 3 out of range"):
        exact_average_delta(iter([trained_adapter(1)]), [1.0], layer=3)


# ---- SVD output rank ----------------------------------------------------------


@pytest.mark.parametrize("rank", [0, -1, 33])
def test_truncated_svd_rank_bounds(rank):
    with pytest.raises(ValueError, match="rank must be in"):
        truncated_svd_refactor(np.ones((32, 32)), rank)


def test_changing_the_output_rank_preserves_the_effective_update():
    """alpha scales with the output rank so (alpha/r) * B @ A is unchanged.
    Before the review fix alpha was copied, silently rescaling the update."""
    ads = [trained_adapter(s, dim=16, rank=4) for s in (1, 2)]
    ads = [_with_alpha(a, 8.0) for a in ads]  # scaling 2.0
    w = [1.0, 3.0]
    exact = exact_average_delta(iter(ads), w)
    merged = aggregate_svd(iter(ads), w, rank=8)  # mean of two rank-4 maps has rank <= 8
    assert merged.rank == 8 and merged.alpha == pytest.approx(16.0)
    assert merged.scaling == pytest.approx(ads[0].scaling)
    for m in merged.target_modules:
        np.testing.assert_allclose(
            merged.scaling * merged.delta_w(m), ads[0].scaling * exact[m], atol=1e-5
        )


def test_same_rank_leaves_alpha_untouched():
    ads = [trained_adapter(s) for s in (1, 2)]
    assert aggregate_svd(iter(ads), [1.0, 1.0]).alpha == ads[0].alpha


# ---- HTTP: incompatible uploads / impossible rank ---------------------------


def test_http_upload_with_mismatching_alpha_is_rejected_at_intake(_clean_server):
    ok = client.post("/clusters/t/uploads", json=upload_body("c1", trained_adapter(1)))
    assert ok.status_code == 201
    r = client.post(
        "/clusters/t/uploads", json=upload_body("c2", _with_alpha(trained_adapter(2), 32.0))
    )
    assert r.status_code == 422 and "share alpha" in r.json()["detail"]
    assert server._clusters["t"].uploads.keys() == {"c1"}  # c2 never buffered


def test_http_a_client_may_replace_its_own_upload_with_a_new_structure(_clean_server):
    client.post("/clusters/t/uploads", json=upload_body("c1", trained_adapter(1)))
    r = client.post(
        "/clusters/t/uploads", json=upload_body("c1", _with_alpha(trained_adapter(2), 32.0))
    )
    assert r.status_code == 201  # only one buffered upload: nothing to disagree with


def test_http_impossible_rank_is_a_422_and_keeps_the_buffer(_clean_server):
    client.post("/clusters/t/uploads", json=upload_body("c1", trained_adapter(1)))
    r = client.post("/clusters/t/aggregate", json={"rank": 999})
    assert r.status_code == 422 and "cannot aggregate" in r.json()["detail"]
    assert server._clusters["t"].uploads.keys() == {"c1"}
    assert server._clusters["t"].round_id == 0
    assert client.post("/clusters/t/aggregate").status_code == 200  # still usable
