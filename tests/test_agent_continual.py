# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for ConsciousAgent's enable_continual=True path.

Drives the Phase-2 local-rule learners (LSM RLS, FEP rank-1, Tsetlin
Type-I) off agent.tell() and verifies their weights actually evolve.
"""


from rain.agent import ConsciousAgent


def test_continual_disabled_by_default():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0)
    assert agent.lsm is None
    assert agent.fep is None
    assert agent.tsetlin is None
    snap = agent.continual_state_snapshot()
    assert snap == {"enabled": False}


def test_continual_enabled_initializes_modules():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
    assert agent.lsm is not None
    assert agent.fep is not None
    assert agent.tsetlin is not None
    snap = agent.continual_state_snapshot()
    assert snap["enabled"] is True
    # LSM, FEP, Tsetlin keys present.
    assert "lsm" in snap and "fep" in snap and "tsetlin" in snap


def test_tell_drives_local_rule_updates():
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
    snap_before = agent.continual_state_snapshot()
    for i in range(15):
        agent.tell(f"sub_{i}", "isa", f"obj_{i}")
    snap_after = agent.continual_state_snapshot()
    # At least one of the modules' parameter L2 norms should change.
    lsm_changed = snap_after["lsm"]["W_out_l2"] != snap_before["lsm"]["W_out_l2"]
    fep_changed = snap_after["fep"]["U_l2"] != snap_before["fep"]["U_l2"]
    tsetlin_changed = (
        snap_after["tsetlin"]["active_inclusions"] != snap_before["tsetlin"]["active_inclusions"]
    )
    assert lsm_changed and fep_changed and tsetlin_changed


def test_kb_writes_still_happen_when_continual_enabled():
    """Continual learning MUST NOT break the KB write path."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
    agent.tell("lion", "lives_in", "savanna")
    ans = agent.ask("lion", "lives_in")
    assert ans.inference_source == "direct"
    assert "savanna" in ans.text


def test_continual_disabled_skip_local_updates():
    """When continual is OFF, tell() doesn't touch any cognitive module
    (because they're None) -- guard against accidental state mutation."""
    agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=False)
    agent.tell("x", "y", "z")
    assert agent.lsm is None
    assert agent.fep is None
    assert agent.tsetlin is None
