# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for RAIN-Net v0.1: HV substrate, encoder bank, MoA router,
hierarchical memory, verifier head, symbolic verifier, and the full
RainNet composition.

These are the green-light tests. If they all pass, the v0.1 architecture
works end-to-end at small scale on a single workstation.
"""

import numpy as np
import pytest

from rain.core.encoder_bank import EncoderBank
from rain.core.hierarchical_memory import (
    EpisodicMemory,
    HierarchicalMemory,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from rain.core.hv_substrate import (
    DEFAULT_DIM,
    Codebook,
    bind,
    bipolarize,
    bundle,
    bundle_weighted,
    encode_kv_map,
    encode_sequence,
    encode_triple,
    hash_to_hv,
    permute,
    query_kv,
    random_hv,
    similarity,
    similarity_matrix,
    unbind,
)
from rain.core.moa_router import MoARouter, default_expert_bank
from rain.core.rain_net import RainNet, RainNetConfig
from rain.core.symbolic_verifier import (
    AuditReport,
    binding_coherence,
    make_audit_report,
)
from rain.core.verifier_head import HVVerifierHead, sample_and_select, self_consistency_vote


# Small dim for fast tests; production uses 10000.
TEST_DIM = 1024


# ============================================================
# HV substrate
# ============================================================


def test_bind_is_self_inverse():
    a = random_hv(TEST_DIM, seed=1)
    b = random_hv(TEST_DIM, seed=2)
    c = bind(a, b)
    recovered = unbind(c, b)
    assert similarity(recovered, a) > 0.99


def test_bundle_preserves_similarity():
    a = random_hv(TEST_DIM, seed=1)
    b = random_hv(TEST_DIM, seed=2)
    s = bundle(a, b)
    # Bundle should be similar to each ingredient
    assert similarity(s, a) > 0.3
    assert similarity(s, b) > 0.3


def test_bundle_weighted_basic():
    a = random_hv(TEST_DIM, seed=1)
    b = random_hv(TEST_DIM, seed=2)
    # Heavy weight on a should make result more similar to a
    s = bundle_weighted([a, b], [0.9, 0.1])
    assert similarity(s, a) > similarity(s, b)


def test_permute_rotates():
    a = random_hv(TEST_DIM, seed=1)
    p1 = permute(a, 1)
    p2 = permute(a, 2)
    # Different shifts should be roughly orthogonal
    assert abs(similarity(p1, p2)) < 0.2
    # Same shift twice == roll by 2
    assert np.array_equal(permute(p1, 1), permute(a, 2))


def test_random_orthogonality():
    # Random HVs at high D should be near-orthogonal
    a = random_hv(TEST_DIM, seed=1)
    b = random_hv(TEST_DIM, seed=2)
    assert abs(similarity(a, b)) < 0.15


def test_hash_to_hv_is_deterministic():
    a = hash_to_hv("hello", dim=TEST_DIM)
    b = hash_to_hv("hello", dim=TEST_DIM)
    assert np.array_equal(a, b)
    c = hash_to_hv("world", dim=TEST_DIM)
    assert not np.array_equal(a, c)


def test_codebook_add_get_cleanup():
    cb = Codebook(dim=TEST_DIM)
    a = cb.add("apple")
    b = cb.add("banana")
    assert not np.array_equal(a, b)
    # Cleanup: noisy version of a should resolve to a
    noisy = bipolarize(a + 0.1 * random_hv(TEST_DIM, seed=99))
    top = cb.cleanup(noisy, top_k=1)
    assert top[0][0] == "apple"


def test_encode_triple_roundtrip():
    cb = Codebook(dim=TEST_DIM)
    s = cb.add("apollo11")
    r = cb.add("landed_on")
    o = cb.add("moon")
    triple = encode_triple(s, r, o)
    # Triple should NOT be similar to a random fact
    other = encode_triple(cb.add("hubble"), cb.add("orbits"), cb.add("earth"))
    assert similarity(triple, other) < 0.3


def test_encode_sequence_distinguishes_order():
    cb = Codebook(dim=TEST_DIM)
    a, b, c = cb.add("x"), cb.add("y"), cb.add("z")
    seq1 = encode_sequence([a, b, c])
    seq2 = encode_sequence([c, b, a])
    # Different orderings should give different HVs
    assert similarity(seq1, seq2) < 0.5


def test_kv_map_query_recovers_value():
    k1 = random_hv(TEST_DIM, seed=10)
    v1 = random_hv(TEST_DIM, seed=11)
    k2 = random_hv(TEST_DIM, seed=12)
    v2 = random_hv(TEST_DIM, seed=13)
    m = encode_kv_map([(k1, v1), (k2, v2)])
    # Querying k1 should recover something more similar to v1 than to v2
    recovered = query_kv(m, k1)
    assert similarity(recovered, v1) > similarity(recovered, v2)


def test_similarity_matrix_batched():
    q = random_hv(TEST_DIM, seed=1)[np.newaxis, :]
    k = np.stack([random_hv(TEST_DIM, seed=i) for i in range(5)], axis=0)
    sims = similarity_matrix(q, k)
    assert sims.shape == (1, 5)


# ============================================================
# Encoder bank
# ============================================================


def test_encoder_bank_text():
    bank = EncoderBank(dim=TEST_DIM)
    h = bank.encode("text", "hello world")
    assert h.shape == (TEST_DIM,)
    assert set(np.unique(np.sign(h)).tolist()) <= {-1.0, 1.0}


def test_encoder_bank_text_consistency():
    bank = EncoderBank(dim=TEST_DIM)
    h1 = bank.encode("text", "the cat sat")
    h2 = bank.encode("text", "the cat sat")
    assert np.array_equal(h1, h2)


def test_encoder_bank_code_differs_from_text():
    bank = EncoderBank(dim=TEST_DIM)
    t = bank.encode("text", "for x in range")
    c = bank.encode("code", "for x in range(10):")
    assert similarity(t, c) < 0.5  # different modalities should cluster apart


def test_encoder_bank_image():
    bank = EncoderBank(dim=TEST_DIM)
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    h = bank.encode("image", img)
    assert h.shape == (TEST_DIM,)


def test_encoder_bank_multi_modal_bind():
    bank = EncoderBank(dim=TEST_DIM)
    parts = [
        ("text", "what is in this picture"),
        ("image", np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)),
    ]
    mm = bank.encode_multi(parts)
    assert mm.shape == (TEST_DIM,)


def test_encoder_bank_register_custom():
    bank = EncoderBank(dim=TEST_DIM)
    bank.register("lidar", lambda p, d: random_hv(d, seed=hash(str(p)) & 0xFFFF))
    h = bank.encode("lidar", [1, 2, 3])
    assert h.shape == (TEST_DIM,)
    assert "lidar" in bank.list_modalities()


def test_encoder_bank_unknown_modality_raises():
    bank = EncoderBank(dim=TEST_DIM)
    with pytest.raises(ValueError, match="unknown modality"):
        bank.encode("xray", [1, 2, 3])


# ============================================================
# MoA router
# ============================================================


def test_router_route_returns_top_k():
    experts = default_expert_bank(dim=TEST_DIM)
    router = MoARouter(experts=experts, top_k=3)
    q = random_hv(TEST_DIM, seed=1)
    routing = router.route(q)
    assert len(routing) == 3
    # weights should sum to ~1.0
    total_w = sum(w for _, w in routing)
    assert abs(total_w - 1.0) < 1e-5


def test_router_forward_produces_hv_and_provenance():
    experts = default_expert_bank(dim=TEST_DIM)
    router = MoARouter(experts=experts, top_k=2)
    q = random_hv(TEST_DIM, seed=1)
    ans, prov = router.forward(q)
    assert ans.shape == (TEST_DIM,)
    assert len(prov) == 2


def test_router_top_k_validation():
    experts = default_expert_bank(dim=TEST_DIM)
    with pytest.raises(ValueError, match="top_k"):
        MoARouter(experts=experts, top_k=99)
    with pytest.raises(ValueError, match="top_k"):
        MoARouter(experts=experts, top_k=0)


def test_router_empty_experts_raises():
    with pytest.raises(ValueError, match="empty"):
        MoARouter(experts=[], top_k=1)


def test_router_records_outcome():
    experts = default_expert_bank(dim=TEST_DIM)
    router = MoARouter(experts=experts, top_k=2)
    q = random_hv(TEST_DIM, seed=1)
    _, prov = router.forward(q)
    router.record_outcome(prov, success=True)
    stats = router.stats()
    # At least one expert should have a success recorded
    assert sum(s["success"] for s in stats) >= 1


def test_router_nudge_domains_updates():
    experts = default_expert_bank(dim=TEST_DIM)
    router = MoARouter(experts=experts, top_k=2)
    q = random_hv(TEST_DIM, seed=1)
    _, prov = router.forward(q)
    orig_domain = experts[0].domain_hv.copy()
    router.nudge_domains(q, prov, success=True, eta=0.5)
    # Some domain HV should have changed (the routed ones)
    changed = any(
        not np.array_equal(experts[i].domain_hv, orig_domain) for i, _ in enumerate(experts)
    )
    assert changed


# ============================================================
# Hierarchical memory
# ============================================================


def test_working_memory_read_write():
    wm = WorkingMemory(capacity=4, dim=TEST_DIM)
    a = random_hv(TEST_DIM, seed=1)
    b = random_hv(TEST_DIM, seed=2)
    wm.write("a", a)
    wm.write("b", b)
    results = wm.read(a, top_k=2)
    assert results[0][0] == "a"  # most similar to a is a


def test_working_memory_capacity_bound():
    wm = WorkingMemory(capacity=2, dim=TEST_DIM)
    for i in range(5):
        wm.write(f"v{i}", random_hv(TEST_DIM, seed=i))
    assert len(wm) == 2


def test_episodic_memory_search():
    em = EpisodicMemory(capacity=8, dim=TEST_DIM)
    q = random_hv(TEST_DIM, seed=1)
    a = random_hv(TEST_DIM, seed=2)
    em.record(q, a)
    results = em.search(q, top_k=1)
    assert len(results) == 1
    assert results[0][0] > 0.99  # exact match


def test_semantic_memory_add_search():
    sm = SemanticMemory(dim=TEST_DIM)
    h = random_hv(TEST_DIM, seed=1)
    sm.add("f1", h, text="Apollo 11 landed on the Moon in 1969.")
    results = sm.search(h, top_k=1)
    assert results[0][1].fact_id == "f1"


def test_semantic_memory_dispute_filters():
    sm = SemanticMemory(dim=TEST_DIM)
    h = random_hv(TEST_DIM, seed=1)
    sm.add("f1", h, text="a")
    sm.dispute("f1")
    results = sm.search(h, top_k=1, include_disputed=False)
    assert len(results) == 0
    results2 = sm.search(h, top_k=1, include_disputed=True)
    assert len(results2) == 1


def test_procedural_memory_threshold():
    pm = ProceduralMemory(dim=TEST_DIM)
    h = random_hv(TEST_DIM, seed=1)
    pm.add("s1", h, adapter_path="/tmp/s1.bin", description="test")
    # Query with same HV should match
    results = pm.search(h, top_k=1, threshold=0.5)
    assert len(results) == 1
    # Query with orthogonal HV should NOT match
    other = random_hv(TEST_DIM, seed=999)
    results2 = pm.search(other, top_k=1, threshold=0.5)
    assert len(results2) == 0


def test_hierarchical_memory_query_all():
    hm = HierarchicalMemory(dim=TEST_DIM)
    h = random_hv(TEST_DIM, seed=1)
    hm.semantic.add("f1", h, text="x")
    hm.working.write("w1", h)
    results = hm.query_all(h, top_k=2)
    assert "semantic" in results
    assert "working" in results
    assert "episodic" in results
    assert "procedural" in results


# ============================================================
# Verifier head
# ============================================================


def test_hv_verifier_score_bounded():
    v = HVVerifierHead(dim=TEST_DIM)
    q = random_hv(TEST_DIM, seed=1)
    a = random_hv(TEST_DIM, seed=2)
    score = v.score(q, a)
    assert -1.0 <= score <= 1.0


def test_hv_verifier_update_changes_state():
    v = HVVerifierHead(dim=TEST_DIM)
    q = random_hv(TEST_DIM, seed=1)
    a = random_hv(TEST_DIM, seed=2)
    s_before = v.score(q, a)
    # Update many times in same direction to overcome noise
    for _ in range(20):
        v.update(q, a, +1)
    s_after = v.score(q, a)
    assert s_after > s_before


def test_hv_verifier_invalid_label_raises():
    v = HVVerifierHead(dim=TEST_DIM)
    q = random_hv(TEST_DIM, seed=1)
    a = random_hv(TEST_DIM, seed=2)
    with pytest.raises(ValueError, match="label"):
        v.update(q, a, 0)


def test_sample_and_select_picks_highest():
    v = HVVerifierHead(dim=TEST_DIM)
    q = random_hv(TEST_DIM, seed=1)
    # Train v strongly to prefer a specific answer
    target = random_hv(TEST_DIM, seed=2)
    for _ in range(200):
        v.update(q, target, +1)
    candidates = [random_hv(TEST_DIM, seed=i + 100) for i in range(5)] + [target]
    chosen, idx = sample_and_select(candidates, q, v)
    # Should choose the target (last index) — strongly trained preference
    assert idx == len(candidates) - 1


def test_self_consistency_vote_picks_most_central():
    # Three candidates clustered around one HV + one outlier
    base = random_hv(TEST_DIM, seed=1)
    noise = lambda s: bipolarize(base + 0.2 * random_hv(TEST_DIM, seed=s))
    candidates = [noise(2), noise(3), noise(4), random_hv(TEST_DIM, seed=999)]
    winner, _score = self_consistency_vote(candidates)
    # Outlier (last) should NOT win
    assert not np.array_equal(winner, candidates[-1])


# ============================================================
# Symbolic verifier
# ============================================================


def test_binding_coherence_high_for_aligned():
    a = random_hv(TEST_DIM, seed=1)
    b = random_hv(TEST_DIM, seed=2)
    # Answer = bundle of cited facts -> coherence should be high
    answer = bundle(a, b)
    coh = binding_coherence(answer, [a, b])
    assert coh > 0.5


def test_binding_coherence_low_for_random():
    a = random_hv(TEST_DIM, seed=1)
    b = random_hv(TEST_DIM, seed=2)
    random_answer = random_hv(TEST_DIM, seed=999)
    coh = binding_coherence(random_answer, [a, b])
    assert coh < 0.3


def test_binding_coherence_empty_returns_zero():
    a = random_hv(TEST_DIM, seed=1)
    assert binding_coherence(a, []) == 0.0


def test_make_audit_report_has_required_fields():
    from rain.core.hierarchical_memory import SemanticFact

    a = random_hv(TEST_DIM, seed=1)
    fact = SemanticFact(
        fact_id="f1",
        hv=a,
        text="The cat sat on the mat.",
        source="test",
    )
    report = make_audit_report(
        answer_text="cat sat on mat",
        answer_hv=a,
        cited_facts=[fact],
        verifier_score=0.8,
        provenance=[("samba_lm", 0.7), ("pure_attn", 0.3)],
    )
    assert isinstance(report, AuditReport)
    assert report.confidence > 0.0
    assert len(report.cited_facts) == 1
    text = report.human_format()
    assert "ANSWER" in text
    assert "CITED FACTS" in text


# ============================================================
# RainNet end-to-end composition
# ============================================================


def test_rain_net_creates_with_default_config():
    net = RainNet(config=RainNetConfig(dim=TEST_DIM, n_candidates=2))
    assert net.config.dim == TEST_DIM
    assert net.memory.semantic is not None
    assert net.router is not None


def test_rain_net_ingest_fact_then_retrieve():
    net = RainNet(config=RainNetConfig(dim=TEST_DIM, n_candidates=2))
    fid = net.ingest_fact("Apollo 11 landed on the Moon in 1969.", source="nasa.gov")
    assert fid.startswith("f")
    report = net.answer("When did Apollo land on the Moon?")
    # Should retrieve the apollo fact
    assert any("Apollo" in f.text for f in report.cited_facts)


def test_rain_net_answer_returns_audit_report():
    net = RainNet(config=RainNetConfig(dim=TEST_DIM, n_candidates=2))
    net.ingest_fact("Python is a programming language.")
    report = net.answer("What is Python?")
    assert isinstance(report, AuditReport)
    assert len(report.provenance) > 0


def test_rain_net_handles_no_facts_gracefully():
    net = RainNet(config=RainNetConfig(dim=TEST_DIM, n_candidates=2))
    report = net.answer("Random question with no KB.")
    assert "don't have grounded information" in report.answer_text
    assert "no facts cited" in " ".join(report.warnings)


def test_rain_net_continual_learning_no_forget():
    """Adding new facts must not affect retrieval of old facts.

    Uses RainNetConfig.semantic_top_k=10 so we check whether the apples
    fact survives in the top-10, not whether it's still top-1 (the v0.1
    text encoder is whitespace-bound; exact lexical overlap matters more
    than semantic similarity).
    """
    net = RainNet(
        config=RainNetConfig(
            dim=TEST_DIM, n_candidates=2, semantic_top_k=10
        )
    )
    apples_id = net.ingest_fact("the first fact about apples is delicious")
    # Add 50 more facts with completely disjoint vocabulary
    for i in range(50):
        net.ingest_fact(f"zebra parade quantum widget {i} marshmallow zeppelin")
    report = net.answer("the first fact about apples")
    # Apples fact must still be retrievable in top-10 after 50 new facts
    fact_ids = [f.fact_id for f in report.cited_facts]
    assert apples_id in fact_ids, f"apples fact {apples_id} not in {fact_ids}"


def test_rain_net_stats_snapshot():
    net = RainNet(config=RainNetConfig(dim=TEST_DIM, n_candidates=2))
    net.ingest_fact("test fact one")
    net.ingest_fact("test fact two")
    s = net.stats()
    assert s["memory"]["semantic"] == 2
    assert "samba_lm" in [r["expert"] for r in s["router"]]


def test_rain_net_feedback_updates_verifier():
    net = RainNet(config=RainNetConfig(dim=TEST_DIM, n_candidates=2))
    net.ingest_fact("test fact")
    report = net.answer("test query")
    # The internal answer HV was computed; we feed back
    a = random_hv(TEST_DIM, seed=42)
    n_before = net.verifier.n_updates
    net.feedback("test query", a, correct=True)
    assert net.verifier.n_updates == n_before + 1


def test_rain_net_ingest_skill():
    net = RainNet(config=RainNetConfig(dim=TEST_DIM, n_candidates=2))
    net.ingest_skill(
        skill_id="math_solver",
        description="solves arithmetic",
        adapter_path="/tmp/math_solver.bin",
        sample_queries=["what is 2+2", "compute 5*7", "evaluate 100/4"],
    )
    assert net.memory.procedural._skills["math_solver"] is not None
