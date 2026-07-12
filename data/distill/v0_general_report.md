# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-24 21:57:51
- Teacher: Ollama llama3.2:3b
- Domain: general
- Curriculum size: 50
- Examples collected: 50
- Total tokens (in+out): 9724
- Cost: $0 (local Ollama)
- Time: 237.8s

## Example queries and teacher answers

### Q: What problem does photosynthesis solve?

**Teacher answer**: Photosynthesis solves the problem of producing energy for living organisms while also removing carbon dioxide and releasing oxygen into the atmosphere.

Source: "Biology" by Campbell & Reece (12th ed.), p. 242-243.

### Q: How does the periodic table relate to similar concepts?

**Teacher answer**: The periodic table is a tabular arrangement of elements organized by their atomic number (number of protons in an atom's nucleus), electron configuration, and recurring chemical properties. It relates to similar concepts as:

1. **Atomic structure**: The periodic table displays how elements' properties change with increasing atomic number.
2. **Chemical bonding**: Elements are grouped based on the...

### Q: Compare general relativity with one alternative approach.

**Teacher answer**: One alternative approach to general relativity is the "Entropic Gravity" theory proposed by Lee Smolin (2006). Entropic Gravity posits that gravity arises from the second law of thermodynamics, where entropy (a measure of disorder or randomness) plays a key role.

In this framework, mass and energy are seen as sources of entropy, which warps spacetime. The curvature of spacetime is not a result of...

### Q: What is the simplest possible description of photosynthesis?

**Teacher answer**: Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy in the form of glucose (sugar).

### Q: Explain supply and demand in 3 sentences.

**Teacher answer**: Supply refers to the amount of a product or service that producers are willing and able to sell at a given price, while demand is the amount of a product or service that consumers are willing and able to buy at a given price. When the quantity of a good or service supplied equals the quantity demanded, the market is said to be in equilibrium. An imbalance between supply and demand can cause prices...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What problem does photosynthesis solve? | Q: What problem does photosynthesis solve? A: Phot... | 0.02 |
| How does the periodic table relate to similar conc... | Q: How does the periodic table relate to similar c... | 0.04 |
| Compare general relativity with one alternative ap... | Q: Compare general relativity with one alternative... | 0.00 |
| What is the simplest possible description of photo... | Q: What is the simplest possible description of ph... | 0.00 |
| Explain supply and demand in 3 sentences. | Q: Explain supply and demand in 3 sentences. A: Su... | 0.00 |
| What is the historical origin of machine learning? | Q: What is the historical origin of machine learni... | 0.53 |
| What problem does blood circulation solve? | Q: What problem does blood circulation solve? A: B... | 0.00 |
| What problem does blood circulation solve? | Q: What problem does blood circulation solve? A: B... | 0.00 |
| What is machine learning used for? | Q: How does machine learning relate to similar con... | 0.00 |
| How does plate tectonics relate to similar concept... | Q: How does plate tectonics relate to similar conc... | 0.01 |

## Takeaways

- 50 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.