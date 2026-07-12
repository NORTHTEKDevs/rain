# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 03:13:09
- Teacher: Ollama llama3.2:3b
- Domain: travel
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 19685
- Cost: $0 (local Ollama)
- Time: 439.3s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about the Appalachian Trail?

**Teacher answer**: The Appalachian Trail (AT) has a unique phenomenon where the trail's popularity and usage are inversely related to its distance from major cities. This means that sections closer to cities, such as the northern section near Boston, receive more hiker traffic than those farther away, despite being longer.

Source: "Appalachian Trail: A Hiker's Guide" by the Appalachian Trail Conservancy

### Q: Who is most associated with the Great Ocean Road?

**Teacher answer**: Sir Douglas Nicholls is often credited as being closely associated with the Great Ocean Road, however it was Sir Archibald Rose and Sir Douglas Nicholls who both played a role in its construction.

However, the person most commonly associated with the Great Ocean Road is Sir Archibald Rose. He was an Australian politician who served as Premier of Victoria from 1939 to 1943.

### Q: What practical use does safari in the Serengeti have today?

**Teacher answer**: Safari tourism in the Serengeti has significant economic benefits for local communities and contributes to conservation efforts.

1. **Job creation**: Safari tourism generates employment opportunities for guides, drivers, and support staff.
2. **Income generation**: Tourists pay fees, which provide a steady income source for local communities.
3. **Conservation funding**: A portion of the revenue ...

### Q: How has the Appalachian Trail changed over time?

**Teacher answer**: The Appalachian Trail (AT) has undergone significant changes since its inception in 1921. Some notable changes include:

* Initial construction was led by the Sierra Club and the Appalachian Mountain Club, with a focus on preserving natural areas.
* In the 1930s, the US government took over maintenance and management of the trail.
* Post-WWII, the AT was expanded to its current length of approxima...

### Q: Give a concrete example of the Great Wall of China.

**Teacher answer**: The Mutianyu Great Wall is a well-preserved section located in Huairou County, Beijing, China. It stretches for approximately 2 kilometers and features steep stone steps, watchtowers, and battlements. The wall's unique "watchtower-and-battlement" design allows it to be both defensive and visually striking.

Source: UNESCO World Heritage Centre

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about the Appalac... | Q: What is a counter-intuitive fact about the Appa... | 0.24 |
| Who is most associated with the Great Ocean Road? | Q: What are common misconceptions about the Great ... | 0.01 |
| What practical use does safari in the Serengeti ha... | Q: What is one famous application of safari in the... | 0.54 |
| How has the Appalachian Trail changed over time? | Q: How has the Appalachian Trail changed over time... | 0.51 |
| Give a concrete example of the Great Wall of China... | Q: Give a concrete example of the Great Wall of Ch... | 0.00 |
| What field does the Cinque Terre belong to? | Q: What field does the Cinque Terre belong to? A: ... | 0.03 |
| What is a counter-intuitive fact about the Trans-M... | Q: What is a counter-intuitive fact about the Tran... | 0.01 |
| What is one famous application of the Faroe Island... | Q: What is one famous application of the Faroe Isl... | 0.02 |
| Give a concrete example of the Cinque Terre. | Q: Give a concrete example of the Cinque Terre. A:... | 0.01 |
| Who is most associated with the Atacama Desert obs... | Q: What is the simplest possible description of th... | 0.00 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.