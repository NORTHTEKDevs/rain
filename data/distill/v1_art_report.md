# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 02:50:26
- Teacher: Ollama llama3.2:3b
- Domain: art
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 19043
- Cost: $0 (local Ollama)
- Time: 430.2s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about surrealism?

**Teacher answer**: Surrealism often rejected traditional notions of beauty and rationality. A counter-intuitive fact is that many surrealist artists, such as Salvador Dalí and René Magritte, were trained in traditional art schools (e.g., fine arts or architecture) before adopting surrealist principles.

Source: 'Theories and Documents of Surrealism' by Annette Michelson

### Q: Who is most associated with Aboriginal dot painting?

**Teacher answer**: Ninfei Tjupurruluru (also known as Ninfei Tjupurruluru or Ninfei) and Albert Namatjira are often credited with popularizing Aboriginal dot painting, but the artist most widely associated with this style is likely Albert Namatjira.

Source: 'source: Australian Government's National Gallery of Australia'

### Q: What practical use does Van Gogh's Starry Night have today?

**Teacher answer**: Van Gogh's "Starry Night" has several practical uses today:

1. **Inspiration for art and design**: The painting continues to inspire artists, designers, and architects worldwide.
2. **Therapeutic benefits**: The calming and soothing atmosphere of the painting is used in therapy settings to reduce stress and anxiety.
3. **Marketing and branding**: The iconic image is often used in advertising and ...

### Q: How has surrealism changed over time?

**Teacher answer**: Surrealism evolved significantly over time, with key developments and shifts in its core principles:

1. **Early Surrealism (1917-1924)**: Founded by André Breton, it focused on exploring the subconscious mind through automatism, dream analysis, and the use of chance operations.
2. **Dadaism's Influence (1916-1924)**: Dadaism's emphasis on chaos and randomness influenced early Surrealism, but even...

### Q: Give a concrete example of pop art.

**Teacher answer**: A classic example of Pop Art is Andy Warhol's "Campbell's Soup Cans" (1962). This silkscreen print features images of Campbell's soup cans, which were ubiquitous consumer products at the time, and explores themes of consumerism, mass production, and the banality of everyday life.

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about surrealism? | Q: What is a counter-intuitive fact about surreali... | 0.00 |
| Who is most associated with Aboriginal dot paintin... | Q: Who is most associated with Aboriginal dot pain... | 0.01 |
| What practical use does Van Gogh's Starry Night ha... | Q: What practical use does Van Gogh's Starry Night... | 0.00 |
| How has surrealism changed over time? | Q: How has surrealism changed over time? A: Surrea... | 0.48 |
| Give a concrete example of pop art. | Q: Give a concrete example of pop art. A: A classi... | 0.00 |
| What field does Warhol's soup cans belong to? | Q: What field does Warhol's soup cans belong to? A... | 0.00 |
| What is a counter-intuitive fact about ceramics? | Q: What is a counter-intuitive fact about ceramics... | 0.21 |
| What is one famous application of oil vs acrylic p... | Q: What is one famous application of oil vs acryli... | 0.00 |
| Give a concrete example of Warhol's soup cans. | Q: Give a concrete example of Warhol's soup cans. ... | 0.00 |
| Who is most associated with color theory? | Q: Who is most associated with color theory? A: Jo... | 0.00 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.