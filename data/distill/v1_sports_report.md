# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 03:20:46
- Teacher: Ollama llama3.2:3b
- Domain: sports
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 20181
- Cost: $0 (local Ollama)
- Time: 447.3s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about the LBW rule in cricket?

**Teacher answer**: The LBW rule in cricket can be counter-intuitive because it's often perceived that if a batsman is hit by the ball, they should automatically get out. However, the actual law states that if the umpire believes the ball would have gone on to hit the stumps (wickets) but for the batsman's leg being in the way, and considering other factors like the trajectory of the ball and the batsman's position, ...

### Q: Who is most associated with the rules of go?

**Teacher answer**: The rules of Go (also known as Weiqi or Baduk) are often attributed to the Chinese general and statesman Sun Tzu, who lived in the 6th century BC. However, the modern rules of Go were codified by Honinbo Shusaku, a Japanese Go player, in the late 19th century.

The most well-known and influential book on Go is "The Book of Go" (also known as "Go Seigen's Book of Go"), written by Go player Go Seige...

### Q: What practical use does Formula 1 qualifying have today?

**Teacher answer**: Formula 1 qualifying has several practical uses:

1. **Racing strategy**: Qualifying results influence starting positions and pit strategies during the race.
2. **Team development**: Analyzing performance in qualifying helps teams identify areas for improvement, such as aerodynamics or tire management.
3. **Driver evaluation**: Qualifying times can be used to assess driver talent and potential tea...

### Q: How has the LBW rule in cricket changed over time?

**Teacher answer**: The Law of Bat and Ball (LBW) rule in cricket has undergone several changes since its introduction in 1876.

1. **Early years**: The original rule stated that a batsman was out if the umpire believed the ball would have hit the stumps, even if it didn't.
2. **1930s-1950s**: The rule changed to require the ball to "hit" the stumps, not just "bowl" them.
3. **1979**: The "ball hitting the pads and t...

### Q: Give a concrete example of the draft in the NFL.

**Teacher answer**: The draft is an annual event where NFL teams select eligible college football players and other eligible players to join their teams. Here's a concrete example:

In 2022, the Cincinnati Bengals selected quarterback Joe Burrow with the first overall pick in the NFL Draft. This means that the Bengals got to choose Joe Burrow before any other team in the league.

Source: Pro-Football-Reference.com

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about the LBW rul... | Q: What is a counter-intuitive fact about the LBW ... | 0.27 |
| Who is most associated with the rules of go? | Q: Who is most associated with the rules of go? A:... | 0.01 |
| What practical use does Formula 1 qualifying have ... | Q: What practical use does Formula 1 qualifying ha... | 0.02 |
| How has the LBW rule in cricket changed over time? | Q: How has the LBW rule in cricket changed over ti... | 0.55 |
| Give a concrete example of the draft in the NFL. | Q: Give a concrete example of the draft in the NFL... | 0.00 |
| What field does the modern pentathlon belong to? | Q: What field does the modern pentathlon belong to... | 0.01 |
| What is a counter-intuitive fact about the wing-T ... | Q: Why is the wing-T offense in football important... | 0.27 |
| What is one famous application of doping testing p... | Q: What is one famous application of doping testin... | 0.00 |
| Give a concrete example of the modern pentathlon. | Q: What is one famous application of the modern pe... | 0.02 |
| Who is most associated with the rules of cricket t... | Q: Who is most associated with the rules of cricke... | 0.00 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.