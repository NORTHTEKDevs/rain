"""Personality + response styling.

A Personality object configures how RCK phrases its answers without
changing what answers it gives. Three knobs:

  tone:        formal | casual | curious | concise
  hedging:     strict | calibrated | confident
  perspective: first_person | third_person

A personality is just a string-mapping; it sits in front of NLG
templates as a final post-process. No learning involved.
"""
from __future__ import annotations

from dataclasses import dataclass


# Phrase banks per tone for the "I know it's X" pattern.
KNOW_PHRASE = {
    "formal":   "It is established that {x}.",
    "casual":   "Yeah, it's {x}.",
    "curious":  "I believe it's {x}!",
    "concise":  "{x}.",
    "default":  "I know it's {x}.",
}

THINK_PHRASE = {
    "formal":   "The evidence suggests {x}, though confidence is limited.",
    "casual":   "Probably {x}, but I'm not 100%.",
    "curious":  "Hmm, I think {x} -- but I'm not sure.",
    "concise":  "Maybe {x}.",
    "default":  "I think {x}, but I'm not certain.",
}

UNKNOWN_PHRASE = {
    "formal":   "I have no record of that.",
    "casual":   "No idea, sorry.",
    "curious":  "Hmm, I don't know that yet!",
    "concise":  "Unknown.",
    "default":  "I don't know.",
}

GREETING = {
    "formal":   "Good day. How may I assist you?",
    "casual":   "Hey! What's up?",
    "curious":  "Hello! What can I help you explore?",
    "concise":  "Hi.",
    "default":  "Hello.",
}


@dataclass
class Personality:
    tone: str = "default"
    hedging: str = "calibrated"
    perspective: str = "first_person"

    def render_know(self, x: str) -> str:
        return KNOW_PHRASE.get(self.tone, KNOW_PHRASE["default"]).format(x=x)

    def render_think(self, x: str) -> str:
        return THINK_PHRASE.get(self.tone, THINK_PHRASE["default"]).format(x=x)

    def render_unknown(self) -> str:
        return UNKNOWN_PHRASE.get(self.tone, UNKNOWN_PHRASE["default"])

    def greeting(self) -> str:
        return GREETING.get(self.tone, GREETING["default"])

    def render_verbal(self, answer: str | None, confidence: float) -> str:
        """The main entry point for rendering an answer with personality."""
        if answer is None:
            return self.render_unknown()
        if self.hedging == "confident":
            return self.render_know(answer)
        if self.hedging == "strict":
            if confidence >= 0.20:
                return self.render_know(answer)
            return self.render_unknown()
        # calibrated
        if confidence >= 0.20:
            return self.render_know(answer)
        if confidence >= 0.10:
            return self.render_think(answer)
        return self.render_unknown()
