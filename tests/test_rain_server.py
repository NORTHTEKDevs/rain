# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Integration tests for the RAIN HTTP server (no live HYMN / judge)."""

import argparse

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from rain.agent import ConsciousAgent
from scripts.rain_server import _make_routes


def _args(**overrides) -> argparse.Namespace:
    base = {
        "kb": None,
        "checkpoint": None,
        "corpus": None,
        "dim": 128,
        "num_shards": 4,
        "rng_seed": 0,
        "enable_continual": False,
        "use_rag": False,
        "rag_max_facts": 3,
        "sample_tokens": 20,
        "temperature": 0.8,
        "top_k": 10,
        "repetition_penalty": 1.0,
        "judge_model": "",
        "min_confidence": 0.5,
        "host": "127.0.0.1",
        "port": 0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class _ServerCase(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        self.agent = ConsciousAgent(dim=256, num_shards=4, seed=0)
        self.agent.tell("lion", "lives_in", "savanna")
        self.agent.tell("tiger", "lives_in", "jungle")
        app = web.Application()
        app.add_routes(_make_routes(self.agent, sampler=None, judge=None, args=_args()))
        return app


class TestHealth(_ServerCase):
    async def test_health_returns_ok(self):
        async with self.client.request("GET", "/health") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert "uptime_seconds" in data


class TestAsk(_ServerCase):
    async def test_ask_kb_hit_returns_citation(self):
        async with self.client.request(
            "POST",
            "/ask",
            json={"subject": "lion", "relation": "lives_in"},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["inference_source"] == "direct"
            assert data["citations"] == [["lion", "lives_in", "savanna"]]
            assert "savanna" in data["text"]

    async def test_ask_missing_body_returns_400(self):
        async with self.client.request(
            "POST",
            "/ask",
            json={"subject": "lion"},
        ) as resp:
            assert resp.status == 400


class TestTell(_ServerCase):
    async def test_tell_then_ask_round_trip(self):
        async with self.client.request(
            "POST",
            "/tell",
            json={"subject": "wolf", "relation": "lives_in", "object": "forest"},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True
        # Now ask should return it
        async with self.client.request(
            "POST",
            "/ask",
            json={"subject": "wolf", "relation": "lives_in"},
        ) as resp:
            data = await resp.json()
            assert "forest" in data["text"]


class TestSampleWithoutSampler(_ServerCase):
    async def test_sample_returns_503_when_no_sampler(self):
        async with self.client.request(
            "POST",
            "/sample",
            json={"prompt": "hello"},
        ) as resp:
            assert resp.status == 503


class _ContinualServerCase(AioHTTPTestCase):
    """Same fixture but with enable_continual=True so the agent ships cognitive_signals."""

    async def get_application(self) -> web.Application:
        self.agent = ConsciousAgent(dim=128, num_shards=4, seed=0, enable_continual=True)
        self.agent.tell("lion", "lives_in", "savanna")
        app = web.Application()
        app.add_routes(
            _make_routes(self.agent, sampler=None, judge=None, args=_args(enable_continual=True))
        )
        return app


class TestAskWithCognitiveSignals(_ContinualServerCase):
    async def test_ask_returns_cognitive_signals(self):
        async with self.client.request(
            "POST",
            "/ask",
            json={"subject": "lion", "relation": "lives_in"},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "cognitive_signals" in data
            cs = data["cognitive_signals"]
            assert "cognitive_agreement" in cs
            assert 0.0 <= cs["cognitive_agreement"] <= 1.0
            assert "fep_cos" in cs
            assert "tsetlin_max_abs_vote" in cs
            assert "lsm_state_l2" in cs


class TestSnapshotAndSelf(_ServerCase):
    async def test_snapshot_returns_disabled_when_continual_off(self):
        async with self.client.request("GET", "/snapshot") as resp:
            data = await resp.json()
            assert data == {"enabled": False}

    async def test_self_returns_text(self):
        async with self.client.request("GET", "/self") as resp:
            data = await resp.json()
            assert isinstance(data["text"], str)
            assert len(data["text"]) > 0


class TestUi(_ServerCase):
    async def test_root_serves_chat_html(self):
        async with self.client.request("GET", "/") as resp:
            assert resp.status == 200
            assert resp.content_type == "text/html"
            body = await resp.text()
            assert "<title>RAIN v0 chat</title>" in body
            # The JS should at least mention the endpoints it calls.
            assert "/ask" in body
            assert "/tell" in body
