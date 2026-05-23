# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""HTTP server exposing ConsciousAgent over JSON endpoints.

Deployment-shape for the v0 RAIN stack. Wraps the chat REPL primitives
(agent.ask/tell/describe, HYMN sampling, calibration tally, optional
LLM judge) as HTTP endpoints so any client can hit RAIN over the wire.

Endpoints:
  GET  /health         -> {"status":"ok", "uptime_seconds": <float>}
  POST /ask            -> body {"subject","relation"} -> Answer JSON
  POST /tell           -> body {"subject","relation","object"} -> {"ok":true}
  POST /describe       -> body {"entity"} -> {"text"}
  POST /sample         -> body {"prompt","n_tokens"=120,"temperature"=0.8}
                          -> {"text"}
  POST /judge          -> body {"subject","relation"} -> verdict JSON
  GET  /tally          -> per-relation calibration tally
  GET  /snapshot       -> continual_state_snapshot diagnostic
  GET  /self           -> agent.self_describe() text

Usage:
    python -m scripts.rain_server \\
        --kb data/kb_seed/llama3b_expanded_filtered.jsonl \\
        --checkpoint data/checkpoints/hymn_carry16_50k.npz \\
        --corpus data/corpora/tiny_shakespeare.txt \\
        --judge-model llama3.2:3b \\
        --use-rag --enable-continual \\
        --host 0.0.0.0 --port 8721

curl -s http://localhost:8721/health
curl -s -X POST http://localhost:8721/ask -H "Content-Type: application/json" \\
    -d '{"subject":"lion","relation":"lives_in"}'
"""

from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from aiohttp import web

from rain.agent import ConsciousAgent
from rain.data.kb_seed import seed_from_jsonl
from rain.feedback.ollama_judge import OllamaJudge, build_triple_prompt
from rain.cognition.rag import KbAugmentedSampler
from scripts.rain_chat import _HymnSampler  # reuse the existing primitive


_STARTED = time.time()


def _make_routes(
    agent: ConsciousAgent,
    sampler: _HymnSampler | None,
    judge: OllamaJudge | None,
    args: argparse.Namespace,
) -> list:
    """Define routes referencing the bound agent + sampler + judge."""

    async def health(_: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "uptime_seconds": round(time.time() - _STARTED, 1),
            "checkpoint": args.checkpoint or None,
            "kb_path": args.kb or None,
            "has_sampler": sampler is not None,
            "has_judge": judge is not None,
            "continual": agent._continual,
        })

    async def ask(req: web.Request) -> web.Response:
        body = await req.json()
        s = body.get("subject")
        r = body.get("relation")
        if not (s and r):
            return web.json_response({"error": "subject and relation required"}, status=400)
        ans = agent.ask(s, r)
        cal = agent.calibration.calibration(r)
        return web.json_response({
            "text": ans.text,
            "epistemic": ans.epistemic,
            "confidence": ans.confidence,
            "inference_source": ans.inference_source,
            "citations": ans.citations,
            "relation_calibration": round(cal, 4),
        })

    async def tell(req: web.Request) -> web.Response:
        body = await req.json()
        s = body.get("subject"); r = body.get("relation"); o = body.get("object")
        if not (s and r and o):
            return web.json_response({"error": "subject, relation, object required"}, status=400)
        agent.tell(s, r, o)
        return web.json_response({"ok": True, "fact": [s, r, o]})

    async def describe(req: web.Request) -> web.Response:
        body = await req.json()
        ent = body.get("entity")
        if not ent:
            return web.json_response({"error": "entity required"}, status=400)
        return web.json_response({"text": agent.describe(ent)})

    async def sample(req: web.Request) -> web.Response:
        if sampler is None:
            return web.json_response({"error": "no HYMN checkpoint loaded"}, status=503)
        body = await req.json()
        prompt = body.get("prompt", "")
        n_tokens = int(body.get("n_tokens", args.sample_tokens))
        temp = float(body.get("temperature", args.temperature))
        top_k = int(body.get("top_k", args.top_k))
        rp = float(body.get("repetition_penalty", args.repetition_penalty))
        text = sampler.sample(
            prompt, n_tokens=n_tokens,
            temperature=temp,
            top_k=(top_k if top_k > 0 else None),
            seed=int(body.get("seed", np.random.randint(0, 1 << 31))),
            repetition_penalty=rp,
        )
        return web.json_response({"prompt": prompt, "text": text})

    async def judge_handler(req: web.Request) -> web.Response:
        if judge is None:
            return web.json_response({"error": "no judge model configured"}, status=503)
        body = await req.json()
        s = body.get("subject"); r = body.get("relation")
        if not (s and r):
            return web.json_response({"error": "subject + relation required"}, status=400)
        ans = agent.ask(s, r)
        stored_obj = ans.citations[0][2] if ans.citations else None
        if stored_obj is None:
            return web.json_response({"error": "no KB citation to judge"}, status=404)
        q, judge_a = build_triple_prompt(s, r, stored_obj)
        verdict = judge.judge(q, judge_a)
        if verdict is None:
            return web.json_response({"error": "judge returned unparseable verdict"}, status=502)
        fired = False
        if verdict.confidence >= args.min_confidence:
            agent.feedback(r, verdict.correct)
            fired = True
        return web.json_response({
            "subject": s, "relation": r, "stored_object": stored_obj,
            "correct": verdict.correct,
            "confidence": verdict.confidence,
            "reasoning": verdict.reasoning,
            "feedback_fired": fired,
            "relation_calibration_after": round(agent.calibration.calibration(r), 4),
        })

    async def tally(_: web.Request) -> web.Response:
        out = {rel: round(agent.calibration.calibration(rel), 4)
               for rel in sorted(agent.calibration._total)}
        return web.json_response(out)

    async def snapshot(_: web.Request) -> web.Response:
        return web.json_response(agent.continual_state_snapshot())

    async def self_describe(_: web.Request) -> web.Response:
        return web.json_response({"text": agent.self_describe()})

    async def ui(_: web.Request) -> web.Response:
        """Serve the built-in chat UI at GET /."""
        ui_path = Path(__file__).resolve().parents[1] / "rain" / "webui" / "chat.html"
        if not ui_path.is_file():
            return web.Response(text="(chat.html missing)", status=404)
        return web.Response(text=ui_path.read_text(encoding="utf-8"),
                            content_type="text/html")

    return [
        web.get("/", ui),
        web.get("/health", health),
        web.post("/ask", ask),
        web.post("/tell", tell),
        web.post("/describe", describe),
        web.post("/sample", sample),
        web.post("/judge", judge_handler),
        web.get("/tally", tally),
        web.get("/snapshot", snapshot),
        web.get("/self", self_describe),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="RAIN HTTP server")
    p.add_argument("--kb", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--corpus", default=None,
                   help="training corpus path (for HYMN char vocab fallback)")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--num-shards", type=int, default=32)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--enable-continual", action="store_true")
    p.add_argument("--use-rag", action="store_true")
    p.add_argument("--rag-max-facts", type=int, default=5)
    p.add_argument("--sample-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--judge-model", default="")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8721)
    args = p.parse_args()

    agent = ConsciousAgent(
        dim=args.dim, num_shards=args.num_shards, seed=args.rng_seed,
        enable_continual=args.enable_continual,
    )
    n_loaded = 0
    if args.kb and Path(args.kb).is_file():
        n_loaded = seed_from_jsonl(agent.kb, args.kb)
    sampler = (_HymnSampler(args.checkpoint, args.corpus)
               if args.checkpoint and Path(args.checkpoint).is_file()
               else None)
    if sampler is not None:
        def _agent_sampler(prompt: str, n_tokens: int) -> str:
            return sampler.sample(
                prompt, n_tokens=n_tokens,
                temperature=args.temperature,
                top_k=(args.top_k if args.top_k > 0 else None),
                seed=np.random.randint(0, 1 << 31),
                repetition_penalty=args.repetition_penalty,
            )
        if args.use_rag:
            rag = KbAugmentedSampler(agent.kb, _agent_sampler,
                                     max_facts=args.rag_max_facts)
            agent.attach_hymn_sampler(rag.sample, n_tokens=args.sample_tokens)
        else:
            agent.attach_hymn_sampler(_agent_sampler, n_tokens=args.sample_tokens)

    judge = OllamaJudge(model=args.judge_model) if args.judge_model else None

    app = web.Application()
    app.add_routes(_make_routes(agent, sampler, judge, args))

    print(f"RAIN HTTP server")
    print(f"  KB facts loaded: {n_loaded}")
    print(f"  HYMN checkpoint: {args.checkpoint or '(none)'}")
    print(f"  use_rag: {args.use_rag}")
    print(f"  continual: {args.enable_continual}")
    print(f"  judge: {args.judge_model or '(none)'}")
    print(f"  listening on http://{args.host}:{args.port}")
    print()
    print(f"try:  curl -s http://{args.host}:{args.port}/health")
    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
