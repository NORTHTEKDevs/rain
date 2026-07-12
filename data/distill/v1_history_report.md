# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 02:04:14
- Teacher: Ollama llama3.2:3b
- Domain: history
- Curriculum size: 100
- Examples collected: 82
- Total tokens (in+out): 17561
- Cost: $0 (local Ollama)
- Time: 3309.6s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about the Renaissance?

**Teacher answer**: The Renaissance was not a unified cultural movement across Europe, but rather a collection of regional revivals in art, science, and culture that occurred independently in different countries.

Source: "The Renaissance" by John Hale (2000)

### Q: Who is most associated with the Russian Revolution?

**Teacher answer**: Vladimir Lenin is most associated with the Russian Revolution. He was a key figure in the Bolshevik Party and played a crucial role in the October Revolution of 1917, which overthrew the provisional government and established the Soviet Union.

Source: 'The Oxford Handbook of the History of Communism' by Mark Leopold (2018)

### Q: What practical use does the Cuban Missile Crisis have today?

**Teacher answer**: The Cuban Missile Crisis serves as a cautionary tale and a reminder of the devastating consequences of nuclear war. Its study has practical uses in:

1. **Nuclear deterrence policy**: Understanding the crisis's dynamics informs modern nuclear strategy and arms control negotiations (source: "Cuban Missile Crisis" by John Lewis Gaddis).
2. **International relations and diplomacy**: The crisis highli...

### Q: How has the Renaissance changed over time?

**Teacher answer**: The Renaissance underwent significant transformations over its approximately 300-year span (14th-17th centuries). Here's a brief overview:

1. **Early Renaissance (1400s)**: Characterized by humanism, revival of classical Greek and Roman culture, and emergence of city-states like Florence and Venice.
2. **High Renaissance (1450s-1520s)**: Marked by the rise of powerful patrons like Lorenzo de' Med...

### Q: Give a concrete example of the fall of Constantinople.

**Teacher answer**: The fall of Constantinople occurred on May 29, 1453, when the Ottoman Empire, led by Mehmed II, conquered the city after a 53-day siege. The Ottomans breached the walls using a massive catapult called the "Basilica" and then poured in troops to capture the city.

Source: "The Fall of Constantinople" by Steven Runciman (source: Runciman, S. (1965). The Fall of Constantinople, 1453.)

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about the Renaiss... | Q: What is a counter-intuitive fact about the Rena... | 0.26 |
| Who is most associated with the Russian Revolution... | Q: Who is most associated with the Russian Revolut... | 0.01 |
| What practical use does the Cuban Missile Crisis h... | Q: What practical use does the Cuban Missile Crisi... | 0.00 |
| How has the Renaissance changed over time? | Q: How has the Renaissance changed over time? A: T... | 0.50 |
| Give a concrete example of the fall of Constantino... | Q: Give a concrete example of the fall of Constant... | 0.00 |
| What field does the silk road belong to? | Q: What field does the silk road belong to? A: The... | 0.02 |
| What is a counter-intuitive fact about the Mauryan... | Q: What is a counter-intuitive fact about the Maur... | 0.26 |
| What is one famous application of the Ottoman Empi... | Q: What is one famous application of the Ottoman E... | 0.00 |
| Give a concrete example of the silk road. | Q: What is one famous application of the silk road... | 0.01 |
| Who is most associated with the Inca Empire? | Q: What is the simplest possible description of th... | 0.04 |

## Takeaways

- 82 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.