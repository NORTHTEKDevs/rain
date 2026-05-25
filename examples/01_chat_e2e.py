# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""RAIN end-to-end demo -- walks through the four passing Tier-1 surfaces live.

Run:  python examples/01_chat_e2e.py

Prints to stdout. No external dependencies beyond the rain package.
"""

from __future__ import annotations

from rain.agent import ConsciousAgent
from rain.cognition.compose import CompositionalReasoner
from rain.cognition.theory_of_mind import TheoryOfMind
from rain.core.relational import Codebook


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def demo_basic_tell_ask():
    banner("1. Basic teach + ask")
    agent = ConsciousAgent(dim=2048, num_shards=8, seed=0)
    agent.tell("rome", "capital_of", "italy")
    agent.tell("paris", "capital_of", "france")
    agent.tell("italy", "locatedin", "europe")

    for s, r in [("rome", "capital_of"), ("paris", "capital_of"), ("madrid", "capital_of")]:
        ans = agent.ask(s, r)
        print(f"Q: {s} {r}")
        print(f"   text:         {ans.text}")
        print(f"   epistemic:    {ans.epistemic}")
        print(f"   confidence:   {ans.confidence}")
        print(f"   citations:    {ans.citations}")
        print()


def demo_n5_grounded_transparency():
    banner("2. N5 -- Grounded transparency (every claim cites a KB fact)")
    agent = ConsciousAgent(dim=2048, num_shards=8, seed=0)
    agent.tell("shakespeare", "wrote", "hamlet")
    agent.tell("hamlet", "isa", "play")
    ans = agent.ask("shakespeare", "wrote")
    print("Q: shakespeare wrote ?")
    print(f"   text:         {ans.text}")
    print(f"   citations:    {ans.citations}")
    # Unanswerable
    ans2 = agent.ask("shakespeare", "born_in")
    print("\nQ: shakespeare born_in ?  (not taught)")
    print(f"   text:         {ans2.text}")
    print(f"   epistemic:    {ans2.epistemic}")
    print("   -> RAIN refuses rather than hallucinating.")


def demo_n4_theory_of_mind():
    banner("3. N4 -- Theory of Mind (Sally-Anne)")
    tom = TheoryOfMind(dim=1024, num_shards=4, seed=0)
    # Sally and Anne both see the marble in the basket
    tom.believe("sally", "marble", "location", "basket")
    tom.believe("anne", "marble", "location", "basket")
    # Sally leaves. Anne moves it. Only Anne sees the move.
    tom.believe("anne", "marble", "location", "box")
    print("Setup: Sally + Anne saw the marble in the basket.")
    print("       Sally leaves. Anne moves it to the box.")
    print()
    print(f"Sally still believes the marble is in the {tom.query_belief('sally', 'marble', 'location')!r}")
    print(f"Anne now believes the marble is in the {tom.query_belief('anne', 'marble', 'location')!r}")
    print()
    print("-> Two characters maintain disjoint beliefs. LLMs typically fail this.")


def demo_n3_compositional_generalization():
    banner("4. N3 -- Compositional generalization (slot-based VSA composition)")
    cb = Codebook(vocab_size=64, dim=2048, seed=0)
    reasoner = CompositionalReasoner(cb, slot_names=["color", "shape", "size"])
    # Compose an unseen combination, then decode ALL slots jointly via the
    # resonant (explaining-away) readout -- one interference-cancelled call.
    assign = {"color": "magenta", "shape": "hexagon", "size": "tiny"}
    candidates = ["red", "green", "blue", "magenta", "yellow",
                  "circle", "square", "triangle", "hexagon",
                  "small", "medium", "large", "tiny"]
    decoded = reasoner.extract_all(reasoner.compose_sum(assign),
                                   slots=["color", "shape", "size"], candidates=candidates)
    print("Composed: (color=magenta, shape=hexagon, size=tiny)")
    print("Resonant joint readout from one composite HV:")
    for slot in ("color", "shape", "size"):
        print(f"   {slot}: {decoded[slot]}")
    print()

    # Why it matters: under heavy slot crowding the per-slot greedy readout
    # collapses while the resonant joint readout holds.
    crowd = Codebook(vocab_size=512, dim=256, seed=1)
    n_slots = 20
    slots = [f"s{i}" for i in range(n_slots)]
    vals = [f"v{i}" for i in range(4 * n_slots)]        # realistic candidate pool
    cr = CompositionalReasoner(crowd, slot_names=slots)
    assign2 = {s: vals[4 * i] for i, s in enumerate(slots)}
    greedy_ok = sum(cr.extract(cr.compose(assign2), s, vals) == assign2[s] for s in slots)
    resonant = cr.extract_all(cr.compose_sum(assign2), slots, vals)
    resonant_ok = sum(resonant[s] == assign2[s] for s in slots)
    print(f"At {n_slots} slots (D=256): greedy recovers {greedy_ok}/{n_slots}, "
          f"resonant recovers {resonant_ok}/{n_slots}")
    print()
    print("-> Compose any combination; decode every slot jointly. Held-out combos "
          "work, and the resonant readout scales to many more slots.")


def demo_n2_calibrated_uncertainty():
    banner("5. N2 -- Calibrated uncertainty (epistemic class per output)")
    agent = ConsciousAgent(dim=2048, num_shards=8, seed=0)
    agent.tell("water", "boils_at", "100c")
    for s, r in [
        ("water", "boils_at"),       # known
        ("oil", "boils_at"),         # unknown
        ("water", "freezes_at"),     # unknown for this relation
    ]:
        ans = agent.ask(s, r)
        print(f"Q: {s} {r} ?")
        print(f"   epistemic: {ans.epistemic:<10}  confidence: {ans.confidence:.2f}")
        print(f"   text: {ans.text}")
        print()


def demo_introspection():
    banner("6. Bonus -- Introspection (what just happened?)")
    agent = ConsciousAgent(dim=1024, num_shards=4, seed=0)
    agent.tell("dog", "isa", "mammal")
    agent.ask("dog", "isa")
    agent.ask("cat", "isa")  # unknown
    print(agent.what_just_happened())


def demo_self_describe():
    banner("7. Bonus -- Self-description")
    agent = ConsciousAgent(dim=1024, num_shards=4, seed=0)
    print(agent.self_describe())


def main():
    print("RAIN end-to-end demo")
    print()
    print("Each section below demonstrates a capability that distinguishes RAIN")
    print("from LLMs: grounded transparency, structural ToM, compositional gen,")
    print("calibrated uncertainty, and structured introspection.")

    demo_basic_tell_ask()
    demo_n5_grounded_transparency()
    demo_n4_theory_of_mind()
    demo_n3_compositional_generalization()
    demo_n2_calibrated_uncertainty()
    demo_introspection()
    demo_self_describe()

    print()
    print("=" * 70)
    print("Demo complete.")
    print('Run `pytest -q -m "not slow"` for the full 133-test acceptance suite.')
    print("=" * 70)


if __name__ == "__main__":
    main()
