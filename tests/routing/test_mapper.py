"""Tests for the SOFAR-on-VSA SVD mapper."""

import numpy as np
import pytest

from rain.routing.mapper import (
    RoutingMapper,
    compute_routing_directions,
)

D = 256  # toy dim for fast tests
K = 16


def test_svd_shapes():
    rng = np.random.default_rng(0)
    cb = rng.choice([-1, 1], size=(100, D)).astype(np.float32)
    result = compute_routing_directions(cb, role_matrix=None, W_hymn_in=None, k=K)
    assert result.V_T.shape == (K, D)
    assert result.sigma.shape == (K,)
    assert result.U.shape == (100, K)
    assert result.D == D
    assert result.k == K


def test_svd_with_roles_and_hymn():
    rng = np.random.default_rng(1)
    cb = rng.choice([-1, 1], size=(64, D)).astype(np.float32)
    roles = rng.choice([-1, 1], size=(8, D)).astype(np.float32)
    W = rng.standard_normal((D, 32)).astype(np.float32)
    result = compute_routing_directions(cb, role_matrix=roles, W_hymn_in=W, k=K)
    assert result.V_T.shape == (K, D)
    # Stacked rows: 64 (cb) + 8 (roles) + 32 (W.T) = 104
    assert result.U.shape == (104, K)


def test_dimension_mismatch_raises():
    rng = np.random.default_rng(2)
    cb = rng.choice([-1, 1], size=(10, D)).astype(np.float32)
    roles_wrong = rng.choice([-1, 1], size=(4, D + 1)).astype(np.float32)
    with pytest.raises(ValueError):
        compute_routing_directions(cb, role_matrix=roles_wrong, W_hymn_in=None, k=K)


def test_cache_returns_same_object_for_unchanged_input():
    rng = np.random.default_rng(3)
    cb = rng.choice([-1, 1], size=(32, D)).astype(np.float32)
    mapper = RoutingMapper(k=K)
    r1 = mapper.get(cb)
    r2 = mapper.get(cb)
    assert r1 is r2  # same object - cache hit


def test_cache_recomputes_on_change():
    rng = np.random.default_rng(4)
    cb1 = rng.choice([-1, 1], size=(32, D)).astype(np.float32)
    cb2 = rng.choice([-1, 1], size=(32, D)).astype(np.float32)
    mapper = RoutingMapper(k=K)
    r1 = mapper.get(cb1)
    r2 = mapper.get(cb2)
    assert r1 is not r2  # different object - cache miss
    assert r1.source_hash != r2.source_hash


def test_v_T_rows_are_unit_length():
    rng = np.random.default_rng(5)
    cb = rng.standard_normal((32, D)).astype(np.float32)
    result = compute_routing_directions(cb, role_matrix=None, W_hymn_in=None, k=K)
    norms = np.linalg.norm(result.V_T, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # SVD V is orthonormal
