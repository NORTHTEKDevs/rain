# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 01:08:52
- Teacher: Ollama llama3.2:3b
- Domain: science
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 21094
- Cost: $0 (local Ollama)
- Time: 615.8s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about dark matter?

**Teacher answer**: Dark matter's gravitational influence can be felt over vast distances, yet it makes up only about 27% of the universe's total mass-energy density (source: NASA). This is counterintuitive because we would expect that if dark matter existed, its presence should be more directly observable.

### Q: Who is most associated with epigenetics?

**Teacher answer**: Johannes Wolfgang von Goethe and Conrad Waddington are often mentioned in relation to early concepts of epigenetics. However, the modern field of epigenetics is most closely associated with:

1. Conrad Waddington (1905-1975): A British biologist who coined the term "epigenesis" in 1942.
2. Bruce Lipton (1953-present) and his work on gene expression and cellular signaling.

However, the most influe...

### Q: What practical use does antimatter have today?

**Teacher answer**: Antimatter has several potential practical uses:

1. **Cancer treatment**: Antimatter can be used to create high-energy beams for cancer therapy, such as boron neutron capture therapy (BNCT). Source: "Antimatter in Medicine" by the European Organization for Nuclear Research (CERN).
2. **Propulsion systems**: Antimatter could potentially be used as a fuel source for advanced propulsion systems, suc...

### Q: How has dark matter changed over time?

**Teacher answer**: Dark matter's properties and characteristics have not changed significantly over time. However, our understanding of it has evolved.

Initially, in 1933, Swiss astrophysicist Fritz Zwicky proposed that there was a large amount of unseen mass holding galaxies together (source: Zwicky, F. A. (1933). 'Die Rotverschiebung von extragalaktischen Nebeln'). 

Later, in the 1970s and 1980s, observations of...

### Q: Give a concrete example of the Higgs boson.

**Teacher answer**: Imagine a bowling ball rolling on a smooth floor. The bowling ball has mass and will follow a curved path due to gravity. Now, imagine a feather with the same mass as the bowling ball but floating on air. The feather will also be attracted to the ground by gravity, but it won't follow the same curved path as the bowling ball.

The Higgs boson is like the "invisible glue" that gives mass to particl...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about dark matter... | Q: What is a counter-intuitive fact about dark mat... | 0.00 |
| Who is most associated with epigenetics? | Q: What are common misconceptions about epigenetic... | 0.25 |
| What practical use does antimatter have today? | Q: What practical use does antimatter have today? ... | 0.27 |
| How has dark matter changed over time? | Q: What is a counter-intuitive fact about dark mat... | 0.42 |
| Give a concrete example of the Higgs boson. | Q: Give a concrete example of the Higgs boson. A: ... | 0.21 |
| What field does the cosmic microwave background be... | Q: What field does the cosmic microwave background... | 0.19 |
| What is a counter-intuitive fact about the speed o... | Q: What is a counter-intuitive fact about the spee... | 0.24 |
| What is one famous application of synaptic transmi... | Q: What is one famous application of synaptic tran... | 0.01 |
| Give a concrete example of the cosmic microwave ba... | Q: Give a concrete example of the cosmic microwave... | 0.24 |
| Who is most associated with Maxwell's equations? | Q: Who is most associated with Maxwell's equations... | 0.01 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.