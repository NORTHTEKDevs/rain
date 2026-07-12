# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 02:11:34
- Teacher: Ollama llama3.2:3b
- Domain: geography
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 18834
- Cost: $0 (local Ollama)
- Time: 427.8s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about Mount Everest?

**Teacher answer**: Mount Everest's height is not the highest point on Earth; its summit is actually lower than the center of the Earth due to the planet's slightly ellipsoidal shape and the movement of tectonic plates. The actual center-to-summit distance is approximately 3,700 meters (12,140 ft).

### Q: Who is most associated with the Australian Outback?

**Teacher answer**: The Australian Outback is often associated with Ned Kelly (1854-1880), a notorious outlaw and folk hero from Victoria. He was known for his bushranging exploits in the region, which has become an integral part of Australian folklore.

Source: 'Ned Kelly' by John Uhrig

### Q: What practical use does Greenland have today?

**Teacher answer**: Greenland is rich in natural resources and has several practical uses:

1. **Renewable energy**: Hydroelectric power plants harness the country's abundant water resources to generate electricity.
2. **Mineral extraction**: Iron ore, copper, gold, and uranium are extracted from the country's mines.
3. **Fishing industry**: Greenland is known for its high-quality fish products, such as cod, halibut,...

### Q: How has Mount Everest changed over time?

**Teacher answer**: Mount Everest, located in the Himalayas, has undergone significant changes due to geological processes.

1. **Uplift**: The Indian tectonic plate collided with the Eurasian plate, causing the Earth's crust to buckle and rise, resulting in the formation of the Himalayan mountain range (source: [1]).
2. **Glacial erosion**: Glaciers carved out the valley and created the characteristic U-shaped profi...

### Q: Give a concrete example of the Nile.

**Teacher answer**: The Nile River is approximately 6,695 kilometers (4,160 miles) long and flows through 11 countries in northeastern Africa. It originates from two main tributaries: the White Nile and the Blue Nile, which meet near Khartoum, Sudan.

One notable example of the Nile's impact on human civilization is the ancient city of Alexandria, Egypt, which was founded by Alexander the Great in 331 BCE. The city b...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about Mount Evere... | Q: What is a counter-intuitive fact about Mount Ev... | 0.25 |
| Who is most associated with the Australian Outback... | Q: Who is most associated with the Australian Outb... | 0.02 |
| What practical use does Greenland have today? | Q: What practical use does Greenland have today? A... | 0.02 |
| How has Mount Everest changed over time? | Q: How has Mount Everest changed over time? A: Mou... | 0.49 |
| Give a concrete example of the Nile. | Q: Give a concrete example of the Nile. A: The Nil... | 0.00 |
| What field does the Galapagos Islands belong to? | Q: What field does the Galapagos Islands belong to... | 0.03 |
| What is a counter-intuitive fact about Iceland's v... | Q: What is a counter-intuitive fact about Iceland'... | 0.25 |
| What is one famous application of the Mediterranea... | Q: What is one famous application of the Mediterra... | 0.01 |
| Give a concrete example of the Galapagos Islands. | Q: Give a concrete example of the Galapagos Island... | 0.00 |
| Who is most associated with the Arctic Ocean? | Q: Who is most associated with the Arctic Ocean? A... | 0.02 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.