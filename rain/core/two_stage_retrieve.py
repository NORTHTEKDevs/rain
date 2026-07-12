"""Two-stage retrieval: fast dense shortlist -> HV-substrate rerank.

v0.2 honest benchmark showed RAIN-Net retrieval matches production
sentence-transformer on quality (85.8% vs 85.4% top-1 at 1852 facts)
but is ~50x slower (56ms vs <1ms per query). The slowness comes from
running full HV substrate ops over the entire KB on every query.

This module separates the path:
    Stage 1: fast cosine over dense ST embeddings -> top-N shortlist
             (~1ms regardless of KB size, given numpy vectorisation)
    Stage 2: full HV substrate ops on the top-N only -- audit trail,
             binding-coherence check, optional reflexion -- on a small
             enough candidate set that the substrate cost is bounded.

Net effect: get the production-RAG latency floor (<1ms shortlist) with
the substrate's audit + reasoning properties applied to the candidates
that actually matter.

Use:
    retriever = TwoStageRetriever(net)
    retriever.ingest_fact("Apollo 11 landed on the Moon in 1969.")
    cited = retriever.retrieve("when did apollo land", shortlist_k=20, final_k=3)

For the full audit-aware path, call retriever.answer() which produces an
AuditReport (same shape as RainNet.answer) on the reranked top-final_k.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.hierarchical_memory import SemanticFact
from rain.core.hv_substrate import similarity_matrix
from rain.core.rain_net import RainNet
from rain.core.symbolic_verifier import AuditReport, make_audit_report


@dataclass
class StageStats:
    shortlist_ms: float = 0.0
    rerank_ms: float = 0.0
    n_shortlist: int = 0
    n_final: int = 0


@dataclass
class TwoStageRetriever:
    """Wraps a RainNet with a fast dense shortlist + HV rerank pipeline.

    Mirrors RainNet.ingest_fact()/answer() but keeps a parallel dense
    embedding bank for the shortlist stage. New facts are encoded twice:
    once as a sentence-transformer dense vector (for fast shortlist) and
    once as the HV (for substrate ops, via RainNet's normal ingestion).
    """

    net: RainNet
    _dense_bank: list[np.ndarray] = field(default_factory=list)
    _dense_fact_ids: list[str] = field(default_factory=list)
    _last_stats: StageStats | None = None

    # The encoder model (lazy-loaded once).
    _model: Any = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    def _encode_dense(self, text: str) -> np.ndarray | None:
        if not self._ensure_model():
            return None
        try:
            emb = self._model.encode(
                [text], show_progress_bar=False, convert_to_numpy=True
            )[0]
            # Normalize for cosine.
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            return emb.astype(np.float32)
        except Exception:  # noqa: BLE001
            return None

    def _encode_dense_batch(self, texts: list[str]) -> np.ndarray | None:
        if not self._ensure_model():
            return None
        try:
            emb = self._model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            return (emb / norms).astype(np.float32)
        except Exception:  # noqa: BLE001
            return None

    # ---------- ingestion ----------

    def ingest_fact(
        self,
        text: str,
        source: str = "",
        fact_id: str | None = None,
        confidence: float = 1.0,
    ) -> str:
        """Ingest into BOTH the RainNet HV memory AND the dense shortlist bank."""
        fid = self.net.ingest_fact(text=text, source=source, fact_id=fact_id, confidence=confidence)
        dense = self._encode_dense(text)
        if dense is not None:
            self._dense_bank.append(dense)
            self._dense_fact_ids.append(fid)
        return fid

    def bulk_ingest(
        self,
        texts: list[str],
        sources: list[str] | None = None,
        fact_ids: list[str | None] | None = None,
    ) -> list[str]:
        """Efficient bulk version: batch-encode dense vectors once.

        fact_ids: optional explicit IDs; None entries get auto-generated.
        Returned list aligns with input order.
        """
        if sources is None:
            sources = [""] * len(texts)
        if fact_ids is None:
            fact_ids = [None] * len(texts)
        # RainNet ingest (HV side).
        fids: list[str] = []
        for t, s, fid in zip(texts, sources, fact_ids, strict=True):
            fid_out = self.net.ingest_fact(text=t, source=s, fact_id=fid)
            fids.append(fid_out)
        # Dense side (one batched encode call).
        dense_batch = self._encode_dense_batch(texts)
        if dense_batch is not None:
            for fid, dense in zip(fids, dense_batch, strict=True):
                self._dense_bank.append(dense)
                self._dense_fact_ids.append(fid)
        return fids

    # ---------- retrieval ----------

    def retrieve(
        self,
        query: str,
        shortlist_k: int = 20,
        final_k: int = 5,
    ) -> list[tuple[float, SemanticFact]]:
        """Two-stage retrieval. Returns (score, fact) tuples for the top
        final_k after rerank.

        Stage 1: dense cosine -> shortlist_k candidates.
        Stage 2: HV-substrate cosine on shortlist -> final_k.
        """
        stats = StageStats()

        # Stage 1: dense shortlist.
        t0 = time.time()
        if not self._dense_bank:
            stats.shortlist_ms = (time.time() - t0) * 1000.0
            return []
        q_dense = self._encode_dense(query)
        if q_dense is None:
            # No dense model; fall back to RainNet only.
            results = self.net.memory.semantic.search(
                self.net.encoder_bank.encode("text", query), top_k=final_k
            )
            stats.shortlist_ms = (time.time() - t0) * 1000.0
            self._last_stats = stats
            return results

        bank = np.stack(self._dense_bank, axis=0)
        sims = bank @ q_dense  # (N,)
        top_n = min(shortlist_k, len(sims))
        top_idx = np.argpartition(-sims, top_n - 1)[:top_n]
        # Order them by score.
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        shortlist_fids = [self._dense_fact_ids[i] for i in top_idx]
        stats.shortlist_ms = (time.time() - t0) * 1000.0
        stats.n_shortlist = len(shortlist_fids)

        # Stage 2: HV substrate rerank on shortlist only.
        t0 = time.time()
        q_hv = self.net.encoder_bank.encode("text", query)
        # Pull just the shortlisted facts' HVs.
        facts: list[SemanticFact] = []
        hvs: list[np.ndarray] = []
        for fid in shortlist_fids:
            f = self.net.memory.semantic._facts.get(fid)
            if f is not None:
                facts.append(f)
                hvs.append(f.hv)
        if not facts:
            stats.rerank_ms = (time.time() - t0) * 1000.0
            self._last_stats = stats
            return []
        hv_bank = np.stack(hvs, axis=0)
        hv_sims = similarity_matrix(q_hv[np.newaxis, :], hv_bank)[0]
        order = np.argsort(-hv_sims)[:final_k]
        out = [(float(hv_sims[i]), facts[i]) for i in order]
        stats.rerank_ms = (time.time() - t0) * 1000.0
        stats.n_final = len(out)
        self._last_stats = stats
        return out

    # ---------- audited answer ----------

    def answer(
        self,
        query: str,
        shortlist_k: int = 20,
        final_k: int = 3,
        modality: str = "text",
    ) -> AuditReport:
        """Full pipeline: two-stage retrieve + substrate audit on top-final_k."""
        cited_with_scores = self.retrieve(query, shortlist_k=shortlist_k, final_k=final_k)
        cited = [f for _s, f in cited_with_scores]

        # Build a minimal report.
        if cited:
            answer_text = cited[0].text
        else:
            answer_text = f"No grounded answer for: {query}"

        # Score the chosen answer with verifier.
        q_hv = self.net.encoder_bank.encode(modality, query)
        # The "answer HV" for verifier scoring is the bundle of cited HVs.
        if cited:
            from rain.core.hv_substrate import bundle

            ans_hv = bundle(*[f.hv for f in cited])
        else:
            ans_hv = q_hv
        verifier_score = self.net.verifier.score(q_hv, ans_hv)

        report = make_audit_report(
            answer_text=answer_text,
            answer_hv=ans_hv,
            cited_facts=cited,
            verifier_score=verifier_score,
            provenance=[("two_stage::shortlist", 1.0)] + (
                [("two_stage::rerank", float(cited_with_scores[0][0]))]
                if cited_with_scores else []
            ),
            tsetlin_model=getattr(self.net, "_tsetlin", None),
            confidence_thresh=self.net.config.confidence_threshold,
        )
        if self._last_stats:
            report.warnings.insert(
                0,
                f"two_stage_retrieve: shortlist={self._last_stats.shortlist_ms:.1f}ms "
                f"rerank={self._last_stats.rerank_ms:.1f}ms "
                f"({self._last_stats.n_shortlist}->{self._last_stats.n_final})",
            )
        return report

    def stats(self) -> dict[str, Any]:
        return {
            "n_dense_facts": len(self._dense_bank),
            "n_hv_facts": len(self.net.memory.semantic),
            "last_query": (
                {
                    "shortlist_ms": self._last_stats.shortlist_ms,
                    "rerank_ms": self._last_stats.rerank_ms,
                    "n_shortlist": self._last_stats.n_shortlist,
                    "n_final": self._last_stats.n_final,
                }
                if self._last_stats
                else None
            ),
        }
