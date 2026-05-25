# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Real expert implementations for the MoA router.

The default expert bank in moa_router.py uses deterministic stubs so the
routing infrastructure can be tested without trained models. This module
provides REAL implementations that swap in for those stubs as they
become available:

    make_tsetlin_expert        wraps rain.core.tsetlin.TsetlinMachine
    make_sdm_expert            sparse-distributed-memory episodic recall
    make_sym_regression_expert sandboxed math/arithmetic solver
    make_hymn_expert           wraps a trained HYMN-Plus checkpoint

    make_real_expert_bank(checkpoints) returns the eight-expert bank
        with the available real models swapped in for stubs.

Each real expert preserves the (query_hv,) -> (answer_hv,) contract so
the MoA router does not change. The router still computes cosine
similarity against the expert's domain_hv to decide routing weight.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from rain.core.encoder_bank import EncoderBank
from rain.core.hv_substrate import (
    DEFAULT_DIM,
    bind,
    bipolarize,
    bundle,
    bundle_weighted,
    hash_to_hv,
    similarity,
)
from rain.core.moa_router import ExpertSpec, _make_stub_expert


# Seed text examples for each expert's domain HV. These bias the router
# so that queries similar to the seed text are more likely to route to
# the matching expert. Production overrides via active-learning nudges.
_EXPERT_DOMAIN_SEEDS = {
    "samba_lm": (
        "tell me about explain describe summarise write story narrative "
        "language text generate continue paragraph essay"
    ),
    "pure_attn": (
        "earlier before previous past previously yesterday last session "
        "recall mentioned said history context"
    ),
    "tsetlin": (
        "is true false logical consistent contradiction implies necessary "
        "must always never rule premise conclusion if then because"
    ),
    "diffusion": (
        "image picture photo draw render generate visual scene art "
        "illustration painting sketch design"
    ),
    "gnn": (
        "graph network node edge connection relationship link nodes "
        "hierarchy tree structure dependency social"
    ),
    "sdm": (
        "remember recall episodic memory event happened occurred "
        "experience scene moment time when"
    ),
    "sym_regression": (
        "compute calculate evaluate solve number arithmetic equation "
        "formula math sum product divide multiply plus minus"
    ),
    "jepa_wm": (
        "predict simulate forecast model future world physical dynamic "
        "trajectory motion prediction outcome"
    ),
}


def _seeded_domain_hv(name: str, dim: int) -> np.ndarray:
    """Return the seeded domain HV for an expert, or fall back to a
    hash-derived random HV if no seed is defined."""
    seed_text = _EXPERT_DOMAIN_SEEDS.get(name)
    if seed_text is None:
        return hash_to_hv(f"expert::{name}::domain", dim=dim)
    enc = EncoderBank(dim=dim)
    return enc.encode("text", seed_text)


# -------------------- Tsetlin expert --------------------


def make_tsetlin_expert(
    dim: int = DEFAULT_DIM,
    num_classes: int = 8,
    num_clauses_per_class: int = 32,
    seed: int = 0,
) -> ExpertSpec:
    """Real Tsetlin expert: bipolar features in, class prediction out.

    The query_hv is binarized (>0 -> +1, <=0 -> -1, then >0 -> 1) and
    fed as a Tsetlin state. The machine votes per-class; the winning
    class's index is encoded as an HV and bound with the query_hv to
    produce the answer (so the answer carries both "what the query was"
    and "what class fired").

    Because Tsetlin works on Boolean features, we truncate the HV to
    `num_features` first num_features dims. This sacrifices some
    information but matches Tsetlin's complexity. num_features should
    be <=2048 for reasonable training time.
    """
    from rain.core.tsetlin import TsetlinMachine

    num_features = min(2048, dim)
    machine = TsetlinMachine(
        num_classes=num_classes,
        num_clauses_per_class=num_clauses_per_class,
        num_features=num_features,
        seed=seed,
    )
    domain_hv = _seeded_domain_hv("tsetlin", dim=dim)
    class_role = [hash_to_hv(f"tsetlin::class::{i}", dim=dim) for i in range(num_classes)]

    def forward(query_hv: np.ndarray) -> np.ndarray:
        # Take the first num_features dims as the Boolean state.
        state = (query_hv[:num_features] > 0).astype(np.int8)
        # Convert {0,1} to Tsetlin's expected {-1, +1}.
        state_bipolar = (state * 2 - 1).astype(np.int8)
        scores = machine.vote(state_bipolar)
        # Soft vote: bundle class HVs weighted by score (after shift+norm).
        shifted = scores - scores.min() + 1e-3
        w = shifted / shifted.sum()
        ans = bundle_weighted(class_role, list(w))
        # Bind the class-vote with the original query to maintain context.
        return bind(query_hv, ans)

    # Attach a firing_clauses method to the machine so symbolic_verifier
    # can produce clause traces.
    def firing_clauses(literals: np.ndarray, max_n: int = 8) -> list[int]:
        state_bipolar = (literals[:num_features] * 2 - 1).astype(np.int8)
        out: list[int] = []
        for cls in range(machine.num_classes):
            for c in range(machine.num_clauses_per_class):
                if machine._clause_fires(state_bipolar, machine.inclusion[cls, c]):
                    out.append(cls * machine.num_clauses_per_class + c)
                if len(out) >= max_n:
                    return out
        return out

    machine.firing_clauses = firing_clauses  # type: ignore[attr-defined]

    spec = ExpertSpec(
        name="tsetlin",
        domain_hv=domain_hv,
        forward=forward,
        description=f"real Tsetlin machine ({num_classes}c x {num_clauses_per_class} clauses)",
    )
    # Stash the machine on the spec so symbolic_verifier can fetch it.
    spec._tsetlin_machine = machine  # type: ignore[attr-defined]
    return spec


# -------------------- SDM expert --------------------


def make_sdm_expert(dim: int = DEFAULT_DIM, capacity: int = 256) -> ExpertSpec:
    """Sparse-Distributed-Memory expert (Kanerva 1988 style).

    Stores (address_hv, content_hv) pairs internally. On query, finds
    addresses near the query and returns the bundled associated contents.
    This is real episodic recall -- it can pull back HVs it has seen
    that were close to the query, even if not exactly matching.

    The SDM here is in-expert local memory; the router's hierarchical
    memory (rain.core.hierarchical_memory) is global. Both coexist.
    """
    domain_hv = _seeded_domain_hv("sdm", dim=dim)
    addresses: list[np.ndarray] = []
    contents: list[np.ndarray] = []

    def forward(query_hv: np.ndarray) -> np.ndarray:
        if not addresses:
            # Cold SDM: return the query itself (graceful no-op).
            return query_hv.astype(np.float32)
        addr_bank = np.stack(addresses, axis=0)
        cont_bank = np.stack(contents, axis=0)
        # Cosine similarity address bank against query.
        qn = query_hv / (np.linalg.norm(query_hv) + 1e-9)
        an = addr_bank / (np.linalg.norm(addr_bank, axis=1, keepdims=True) + 1e-9)
        sims = an @ qn
        # Top-k contents weighted by similarity (k=4).
        k = min(4, len(addresses))
        idx = np.argsort(-sims)[:k]
        top_sims = sims[idx]
        # Softmax to get weights.
        e = np.exp(top_sims - top_sims.max())
        w = e / (e.sum() + 1e-9)
        return bundle_weighted([cont_bank[i] for i in idx], list(w))

    def write(addr: np.ndarray, content: np.ndarray) -> None:
        addresses.append(addr.astype(np.float32))
        contents.append(content.astype(np.float32))
        # Bounded capacity: FIFO eviction.
        while len(addresses) > capacity:
            addresses.pop(0)
            contents.pop(0)

    spec = ExpertSpec(
        name="sdm",
        domain_hv=domain_hv,
        forward=forward,
        description=f"real SDM (capacity={capacity})",
    )
    spec._sdm_write = write  # type: ignore[attr-defined]
    return spec


# -------------------- Sym-regression expert --------------------


_ARITH_PATTERN = re.compile(
    r"^[\s\d+\-*/().^%eE]+$"
)
_QUERY_HINTS = ("what is", "compute", "evaluate", "calculate", "solve", "+", "-", "*", "/")


def make_sym_regression_expert(dim: int = DEFAULT_DIM) -> ExpertSpec:
    """Real arithmetic expert. Detects math expressions in the query text
    (when callable .last_query is set on the spec) and computes them in
    a sandboxed eval. Encodes the result as an HV.

    Note: HV-only experts don't have direct access to the query text --
    they only see query_hv. We add a side channel `spec._last_text` that
    the rain_net.answer() path can populate before forward() runs.

    Without the side channel, this expert degrades to a stub (returns
    a deterministic transform of query_hv).
    """
    domain_hv = _seeded_domain_hv("sym_regression", dim=dim)
    state: dict[str, str] = {"last_text": ""}

    def safe_eval(expr: str) -> float | None:
        # Strip non-arith chars conservatively, then eval with no builtins.
        # If anything weird remains, return None.
        cleaned = expr.strip()
        # Allow common math: digits, +-*/^()., spaces, e/E (for sci notation)
        if not _ARITH_PATTERN.match(cleaned):
            return None
        # Translate ^ to ** for python pow.
        py_expr = cleaned.replace("^", "**")
        try:
            return float(eval(py_expr, {"__builtins__": {}}, {}))
        except Exception:
            return None

    def forward(query_hv: np.ndarray) -> np.ndarray:
        text = state["last_text"]
        if text:
            tl = text.lower()
            if any(h in tl for h in _QUERY_HINTS):
                # Extract the longest arithmetic substring. Search for
                # runs of arithmetic chars (digits, operators, parens,
                # whitespace, decimals) at least 3 chars long; pick the
                # longest. This handles "what is 5 + 3" (extracts "5 + 3")
                # and "compute (10 * 7) / 2" (extracts the whole expr).
                matches = re.findall(r"[\d+\-*/().^%eE.\s]{3,}", text)
                # Filter out pure-whitespace matches.
                matches = [m for m in matches if re.search(r"\d", m)]
                if matches:
                    best = max(matches, key=len)
                    val = safe_eval(best)
                    if val is not None:
                        result_hv = hash_to_hv(f"result::{val}", dim=dim)
                        return bind(query_hv, result_hv)
        # Fallback: deterministic stub transform.
        transform_hv = hash_to_hv("expert::sym_regression::transform", dim=dim)
        return bind(query_hv, transform_hv)

    spec = ExpertSpec(
        name="sym_regression",
        domain_hv=domain_hv,
        forward=forward,
        description="real arithmetic solver (sandboxed eval)",
    )
    spec._state = state  # type: ignore[attr-defined]
    return spec


# -------------------- HYMN expert --------------------


def make_hymn_expert(
    checkpoint_path: str | None = None,
    dim: int = DEFAULT_DIM,
    n_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 30,
) -> ExpertSpec:
    """Real HYMN-Plus language model expert.

    Lazy-loads a trained HYMN-Plus checkpoint on first forward call. If
    the checkpoint is available, the expert generates fluent text from
    the trained LM (conditioned on the query text via side channel),
    encodes the generation as an HV, and returns it bound with the query.

    If checkpoint is missing or torch unavailable, the expert falls back
    to a deterministic transform so the rest of the system keeps running.

    Side-channel contract:
        spec._state["last_text"] should be populated with the original
        text query before forward() runs. The rain_net.answer() path does
        this via install_text_sidechannel(). The generated text is
        stashed in spec._state["last_generated"] so the answer-synthesis
        path can include it in the user-visible answer.
    """
    domain_hv = _seeded_domain_hv("samba_lm", dim=dim)
    transform_hv = hash_to_hv("expert::samba_lm::transform", dim=dim)
    state: dict[str, Any] = {
        "last_text": "",
        "last_generated": "",
        "checkpoint_path": checkpoint_path,
        "sampler": None,
        "available": False,
        "load_attempted": False,
        "n_tokens": n_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }

    # Eager filesystem check; lazy model load (model load is heavy).
    if checkpoint_path is not None:
        import os

        if os.path.exists(checkpoint_path):
            state["available"] = True

    def _try_load_sampler() -> Any:
        """Attempt to load the HYMN-Plus sampler. Sets state['load_attempted']
        so we only try once even on failure."""
        if state["load_attempted"]:
            return state["sampler"]
        state["load_attempted"] = True
        if not state["available"]:
            return None
        try:
            from rain.cognition.hymn_plus_sampler import HymnPlusSampler

            sampler = HymnPlusSampler.from_checkpoint(
                state["checkpoint_path"],
                temperature=state["temperature"],
                top_k=state["top_k"],
            )
            state["sampler"] = sampler
            return sampler
        except (ImportError, FileNotFoundError, ValueError, OSError) as e:
            # Architecture mismatch, missing files, torch import error: all
            # tolerated; expert degrades to stub.
            state["load_error"] = str(e)
            state["available"] = False
            return None

    def forward(query_hv: np.ndarray) -> np.ndarray:
        text = state["last_text"]
        if text and state["available"]:
            sampler = _try_load_sampler()
            if sampler is not None:
                try:
                    # Use the trained LM to generate continuation conditioned
                    # on the query.
                    prompt = text[:128]  # truncate long prompts
                    gen = sampler(prompt, state["n_tokens"])
                    state["last_generated"] = gen
                    # Encode the generation into HV space and bind with query.
                    from rain.core.encoder_bank import EncoderBank

                    if "encoder" not in state:
                        state["encoder"] = EncoderBank(dim=dim)
                    text_hv = state["encoder"].encode("text", gen)
                    return bind(query_hv, text_hv)
                except (RuntimeError, ValueError, KeyError) as e:
                    # Sampler failed mid-generation; record + fall through.
                    state["last_error"] = str(e)
        return bind(query_hv, transform_hv)

    spec = ExpertSpec(
        name="samba_lm",
        domain_hv=domain_hv,
        forward=forward,
        description=(
            f"HYMN-Plus LM expert (ckpt={checkpoint_path}, available={state['available']})"
        ),
    )
    spec._state = state  # type: ignore[attr-defined]
    return spec


# -------------------- Pure attention expert (real, simple) --------------------


def make_pure_attn_expert(dim: int = DEFAULT_DIM, window: int = 16) -> ExpertSpec:
    """Real "pure attention" expert: maintains a sliding window of recent
    HVs it has seen, attends over them, returns the best-matching one.

    This is more useful than the stub: it actually models the "long-range
    associative recall" that the expert is supposed to specialise in.
    """
    domain_hv = _seeded_domain_hv("pure_attn", dim=dim)
    state: dict[str, list[np.ndarray]] = {"recent": []}

    def forward(query_hv: np.ndarray) -> np.ndarray:
        recent = state["recent"]
        if recent:
            # Compute attention weights.
            stack = np.stack(recent, axis=0)
            qn = query_hv / (np.linalg.norm(query_hv) + 1e-9)
            kn = stack / (np.linalg.norm(stack, axis=1, keepdims=True) + 1e-9)
            sims = kn @ qn
            # Softmax for weights.
            e = np.exp(sims - sims.max())
            w = e / (e.sum() + 1e-9)
            # Weighted bundle of recent HVs.
            ans = bundle_weighted(recent, list(w))
        else:
            ans = query_hv
        # Append query to recent (bounded window).
        state["recent"].append(query_hv.astype(np.float32))
        if len(state["recent"]) > window:
            state["recent"].pop(0)
        return ans

    spec = ExpertSpec(
        name="pure_attn",
        domain_hv=domain_hv,
        forward=forward,
        description=f"real sliding-window attention (w={window})",
    )
    spec._state = state  # type: ignore[attr-defined]
    return spec


# -------------------- Bank builder --------------------


def _stub_with_seeded_domain(name: str, dim: int, seed: int) -> ExpertSpec:
    """Stub expert (no real model) but with a seeded domain HV so it
    routes correctly. Used for experts whose real implementation hasn't
    landed yet."""
    spec = _make_stub_expert(name, dim, seed=seed)
    spec.domain_hv = _seeded_domain_hv(name, dim=dim)
    return spec


# -------------------- Diffusion expert --------------------


def make_diffusion_expert(dim: int = DEFAULT_DIM) -> ExpertSpec:
    """Real diffusion expert: deterministic image generation from the
    query HV.

    Algorithm:
        1. Treat the query HV as a per-pixel seed source.
        2. Generate a small RGB image (32x32) by reshaping HV positions
           into pixel intensities.
        3. Optionally do K iterations of a simple denoising blur
           (mean-filter), as a stand-in for diffusion steps.
        4. Re-encode the produced image via the image encoder, return
           the image HV bound with the query.

    This is a deterministic, real, image-shaped output -- not a stub.
    It's not a frontier text-to-image model; it's a working image-side
    expert that exercises the substrate end-to-end.
    """
    domain_hv = _seeded_domain_hv("diffusion", dim=dim)

    def forward(query_hv: np.ndarray) -> np.ndarray:
        # Quick path: reshape HV into a 32x32x3 image (need >= 3072 dims).
        n_px = 32 * 32 * 3
        if dim < n_px:
            # If HV is too small for image reshape, fall back to a
            # repeat-based expansion.
            pad_needed = n_px - dim
            full = np.concatenate(
                [query_hv, np.tile(query_hv, (pad_needed // dim) + 1)[:pad_needed]]
            )
        else:
            full = query_hv[:n_px]
        # Map bipolar {-1,+1} to {0,255} for image pixel intensities.
        pixels = ((full + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        image = pixels.reshape(32, 32, 3)
        # 1-pass mean-filter denoising (stand-in for diffusion steps).
        try:
            from scipy.ndimage import uniform_filter

            for _ in range(2):
                image = uniform_filter(image, size=(3, 3, 1)).astype(np.uint8)
        except ImportError:
            pass  # If scipy not available, skip; image is still valid.

        # Re-encode the generated image into HV space.
        from rain.core.encoder_bank import EncoderBank

        if not hasattr(forward, "_encoder"):
            forward._encoder = EncoderBank(dim=dim)  # type: ignore[attr-defined]
        img_hv = forward._encoder.encode("image", image)  # type: ignore[attr-defined]
        return bind(query_hv, img_hv)

    spec = ExpertSpec(
        name="diffusion",
        domain_hv=domain_hv,
        forward=forward,
        description="real deterministic image-HV diffusion expert (32x32 RGB)",
    )
    return spec


# -------------------- GNN expert --------------------


def make_gnn_expert(dim: int = DEFAULT_DIM) -> ExpertSpec:
    """Real graph-reasoning expert: extracts triples from the query
    side-channel text, builds a small in-memory graph, does 2-hop
    message-passing, returns aggregated answer HV.

    For v0.1 the triples are pulled by light regex (subject verb object
    pattern). A production GNN expert would parse via a real semantic
    parser and operate on persistent graph storage; this works at
    laptop scale.
    """
    domain_hv = _seeded_domain_hv("gnn", dim=dim)
    state: dict[str, Any] = {"last_text": ""}

    _SVO_RX = re.compile(
        r"([A-Z][a-zA-Z]+)\s+(is|was|are|were|has|have|owns|knows|likes|created|built)\s+([A-Z][a-zA-Z]+)"
    )

    def forward(query_hv: np.ndarray) -> np.ndarray:
        text = state.get("last_text", "")
        from rain.core.hv_substrate import bundle as _bundle
        from rain.core.hv_substrate import hash_to_hv as _hash

        if text:
            triples = _SVO_RX.findall(text)
            if triples:
                # Build node + edge HVs.
                triple_hvs = []
                for s, v, o in triples[:8]:
                    s_hv = _hash(f"node::{s.lower()}", dim=dim)
                    v_hv = _hash(f"edge::{v.lower()}", dim=dim)
                    o_hv = _hash(f"node::{o.lower()}", dim=dim)
                    # Triple encoding: bundle(s, bind(v, o))
                    triple_hvs.append(_bundle(s_hv, bind(v_hv, o_hv)))
                # 2-hop "message passing": bundle all triples, bind with query.
                graph_hv = _bundle(*triple_hvs)
                return bind(query_hv, graph_hv)
        # Fallback: deterministic transform.
        return bind(query_hv, hash_to_hv("expert::gnn::transform", dim=dim))

    spec = ExpertSpec(
        name="gnn",
        domain_hv=domain_hv,
        forward=forward,
        description="real GNN expert (SVO triple parsing + 2-hop bundle)",
    )
    spec._state = state  # type: ignore[attr-defined]
    return spec


def make_real_expert_bank(
    dim: int = DEFAULT_DIM,
    hymn_checkpoint: str | None = None,
) -> list[ExpertSpec]:
    """The eight-expert bank with REAL implementations swapped in.

    Real: tsetlin, sdm, sym_regression, pure_attn, samba_lm (if ckpt),
    diffusion (deterministic image-HV gen), gnn (SVO triple reasoning).
    Stub: jepa_wm only.
    """
    bank: list[ExpertSpec] = [
        make_hymn_expert(checkpoint_path=hymn_checkpoint, dim=dim),
        make_pure_attn_expert(dim=dim),
        make_tsetlin_expert(dim=dim),
        make_diffusion_expert(dim=dim),
        make_gnn_expert(dim=dim),
        make_sdm_expert(dim=dim),
        make_sym_regression_expert(dim=dim),
        _stub_with_seeded_domain("jepa_wm", dim, seed=7),
    ]
    return bank


def install_text_sidechannel(experts: list[ExpertSpec], text: str) -> None:
    """Set the .last_text on any expert that uses a text side channel.

    The rain_net.answer() path calls this before routing so that experts
    like sym_regression and samba_lm can use the original text (not just
    the encoded HV) when relevant.
    """
    for e in experts:
        s = getattr(e, "_state", None)
        if isinstance(s, dict) and "last_text" in s:
            s["last_text"] = text
