"""Free Energy minimisation and active inference for next-symbol selection.

The agent maintains a small variational generative model:
    p(s_{t+1} | s_t)  parameterised as a low-rank linear map in HV space.

We factor the transition as  A ~= U @ V.T  with U, V shape (D, r) and
r << D.  Updates are O(D * r), memory is O(D * r). At D=2048, r=64 that's
~250K params and ~250K flops per step instead of ~4M.

At each step we:
  1. PERCEIVE: do a single rank-1 update of U from the prediction error.
  2. PLAN: enumerate candidate next-symbols (a small set from cleanup),
     compute the expected free energy, choose argmin.
  3. ACT: emit the chosen symbol.

EFE is now:
    G(a) = || codebook[a] - U @ (V.T @ s_t) ||^2 / (2 sigma^2)
         + freq_prior(a)
         + repetition_penalty(a)

The novelty term from v0.1 is gone -- it was biasing toward rare characters
and making generation gibberish. Instead we use a frequency prior (more
common = lower G = more likely to be chosen) and an explicit
repetition_penalty against recent emissions.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from rck.codebook import Codebook


@dataclass
class ActiveInference:
    """Low-rank linear-Gaussian generative model in HV space + EFE policy.

    Args:
        dim: HV dimensionality.
        rank: rank of the transition factorisation. 32-128 is plenty in practice.
        lr: learning rate for the rank-1 update.
        sigma: observation noise.
        temperature: softmax temperature when stochastic=True.
        freq_prior_weight: weight on -log(freq) prior. 0 disables.
        repetition_penalty: G(a) += repetition_penalty * 1{a in recent}.
        repeat_window: length of recent-emission queue.
    """

    dim: int = 4096
    rank: int = 64
    lr: float = 5e-3
    sigma: float = 1.0
    temperature: float = 0.5
    freq_prior_weight: float = 0.05
    repetition_penalty: float = 0.8
    repeat_window: int = 12
    ngram_repeat_penalty: float = 1.6
    ngram_repeat_window: int = 16

    _U: np.ndarray = field(default=None, init=False)
    _V: np.ndarray = field(default=None, init=False)
    _freq: Counter = field(default_factory=Counter, init=False)
    _recent: deque = field(default=None, init=False)
    _rng: np.random.Generator = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(0)
        # Initialise so that A = U V.T is approximately identity / 2 in expectation.
        scale = np.sqrt(0.5 / self.rank)
        self._U = self._rng.normal(0, scale, (self.dim, self.rank)).astype(np.float32)
        self._V = self._U.copy()  # symmetric init -> A ~ 0.5 * I_low_rank
        self._recent = deque(maxlen=self.repeat_window)

    def predict(self, s: np.ndarray) -> np.ndarray:
        s_f = s.astype(np.float32)
        return self._U @ (self._V.T @ s_f)

    def perceive(self, s: np.ndarray, s_next: np.ndarray) -> float:
        """Rank-1 SGD on U + V to reduce ||s_next - U V.T s||^2."""
        s_f = s.astype(np.float32)
        s_next_f = s_next.astype(np.float32)
        z = self._V.T @ s_f                 # (r,)
        pred = self._U @ z                  # (D,)
        err = s_next_f - pred               # (D,)
        # gradient wrt U: -2 * err outer z. gradient wrt V: -2 * s_f outer (U.T err).
        sn = float(np.dot(s_f, s_f)) + 1e-6
        self._U += self.lr * np.outer(err, z) / sn
        self._V += self.lr * np.outer(s_f, self._U.T @ err) / sn
        return float(np.linalg.norm(err))

    def expected_free_energy(
        self,
        s_t: np.ndarray,
        candidates: dict[Hashable, np.ndarray],
        *,
        signals: dict[str, np.ndarray] | None = None,
        signal_weights: dict[str, float] | None = None,
    ) -> dict[Hashable, float]:
        """G(a) = weighted sum of (negative cosine) similarities to each signal
        source + frequency prior + repetition penalty.

        signals maps a name -> a predicted next-state hypervector (real-valued).
        Default: just the FEP's own transition prediction.
        """
        if signals is None:
            signals = {"fep": self.predict(s_t)}
        if signal_weights is None:
            signal_weights = {k: 1.0 for k in signals}
        # Pre-normalize signal vectors for cosine.
        sig_units: dict[str, np.ndarray] = {}
        for k, v in signals.items():
            v = v.astype(np.float32)
            n = float(np.linalg.norm(v))
            sig_units[k] = v / n if n > 0 else v
        total_freq = sum(self._freq.values()) or 1
        out: dict[Hashable, float] = {}
        for symbol, hv in candidates.items():
            hv_f = hv.astype(np.float32)
            nh = float(np.linalg.norm(hv_f))
            hv_u = hv_f / nh if nh > 0 else hv_f
            # Sum of weighted -cosine across signal sources.
            G = 0.0
            for name, w in signal_weights.items():
                if name not in sig_units:
                    continue
                cos = float(np.dot(hv_u, sig_units[name]))
                G += -w * cos
            # Frequency prior.
            freq = self._freq.get(symbol, 0)
            p = (freq + 1) / (total_freq + len(self._freq) + 1)
            G += -self.freq_prior_weight * float(np.log(p))
            # Anti-repetition: single-char.
            if symbol in self._recent:
                G += self.repetition_penalty
            # Anti-repetition: bigram loop. Detect immediate "ab ab" cycle.
            if len(self._recent) >= 2:
                last_two = (self._recent[-2], self._recent[-1])
                next_two = (self._recent[-1], symbol)
                if last_two == next_two:
                    G += self.ngram_repeat_penalty
            # Anti-repetition: 3-gram loop. Detect "abc abc" cycle.
            if len(self._recent) >= 3:
                last_three = (self._recent[-3], self._recent[-2], self._recent[-1])
                # If emitting `symbol` would mirror what came 3 emissions ago.
                if last_three[-2:] == (self._recent[-1], None):
                    pass
                if (len(self._recent) >= 6 and
                    tuple(list(self._recent)[-6:-3]) == (self._recent[-3], self._recent[-2], self._recent[-1])):
                    G += self.ngram_repeat_penalty
            out[symbol] = G
        return out

    def act(
        self,
        s_t: np.ndarray,
        codebook: Codebook,
        top_k: int = 12,
        stochastic: bool = False,
        bias_candidates: list[Hashable] | None = None,
        teacher_freq_update: Hashable | None = None,
        signals: dict[str, np.ndarray] | None = None,
        signal_weights: dict[str, float] | None = None,
    ) -> tuple[Hashable, dict[Hashable, float]]:
        """Choose the symbol that minimises EFE.

        If signals/signal_weights are supplied, they replace the default
        FEP-only prediction. This is how the agent injects LSM-readout and
        n-gram-query signals into the decision.
        """
        pred = self.predict(s_t)
        nearest = codebook.fast_cleanup(pred, top_k=top_k)
        cand_syms = {sym for sym, _ in nearest}
        if bias_candidates:
            cand_syms.update(bias_candidates)
        if not cand_syms:
            cand_syms = set(codebook.symbols())
        candidates = {sym: codebook.encode(sym) for sym in cand_syms}
        G = self.expected_free_energy(
            s_t, candidates,
            signals=signals, signal_weights=signal_weights,
        )
        if not G:
            return next(iter(codebook.symbols())), {}
        if stochastic:
            chosen = self._sample(G)
        else:
            chosen = min(G, key=G.get)
        # Bookkeeping.
        self._recent.append(chosen)
        if teacher_freq_update is not None:
            self._freq[teacher_freq_update] += 1
        else:
            self._freq[chosen] += 1
        return chosen, G

    def reset_recent(self) -> None:
        self._recent.clear()

    def _sample(self, G: dict[Hashable, float], top_p: float = 0.9) -> Hashable:
        """Top-p (nucleus) softmax sample over -G.

        Picks the smallest set of candidates whose softmax mass exceeds
        `top_p`, renormalises, then samples. Cuts the long tail of bad
        candidates that pure softmax over many low-probability classes
        would otherwise feed in as noise.
        """
        syms = list(G.keys())
        if len(syms) == 1:
            return syms[0]
        T = max(self.temperature, 1e-6)
        scores = -np.array([G[s] for s in syms], dtype=np.float32) / T
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        order = np.argsort(-probs)
        cum = 0.0
        cutoff = len(order)
        for k, i in enumerate(order):
            cum += float(probs[i])
            if cum >= top_p:
                cutoff = k + 1
                break
        kept = order[:cutoff]
        sub = probs[kept]
        sub = sub / sub.sum()
        idx = self._rng.choice(len(kept), p=sub)
        return syms[int(kept[idx])]
