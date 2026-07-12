# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 02:57:47
- Teacher: Ollama llama3.2:3b
- Domain: music
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 19055
- Cost: $0 (local Ollama)
- Time: 430.9s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about the major scale?

**Teacher answer**: The major scale has an "augmented fourth" sound when ascending from C to F# (C-D-E-F#-G-A-B-C). This might seem dissonant or unusual, but it's actually a fundamental property of Western tonal music. Source: source: 'The Musician's Guide to Theory and Analysis' by Jane Piper Clendinning

### Q: Who is most associated with the loudness war?

**Teacher answer**: George Martin and Alan Parsons are often credited with being key figures in the "loudness war," but the person most directly associated with it is likely Bob Ludwig, an engineer who worked extensively with bands like AC/DC, Metallica, and Guns N' Roses.

However, the term 'Loudness War' was popularized by George Martin's work on The Beatles' albums (specifically, Sgt. Pepper's Lonely Hearts Club B...

### Q: What practical use does the British Invasion have today?

**Teacher answer**: The British Invasion's influence can still be seen in various aspects of modern music and popular culture:

1. **Music Industry**: The invasion helped shape the modern music industry, with many iconic bands like The Beatles, The Rolling Stones, and The Who paving the way for future generations of musicians.
2. **Influence on Contemporary Music**: Artists like The Killers, Arctic Monkeys, and Coldp...

### Q: How has the major scale changed over time?

**Teacher answer**: The major scale has undergone significant changes throughout history.

In ancient Greece (5th century BCE), the major scale was based on the harmonic series and consisted of 6 notes, with a specific pattern of whole and half steps: W-W-H-W-W-H. This scale is known as the "Greek major scale" or "Ptolemaic major scale."

Over time, the major scale evolved to include more notes and follow the modern ...

### Q: Give a concrete example of Bach's fugues.

**Teacher answer**: A classic example of Bach's fugue is the "Fugue in G minor" from The Well-Tempered Clavier (Book I), BWV 846.

In this fugue, a theme (subject) is introduced at the beginning and then repeated with variations by different voices, creating a complex interweaving of musical lines. This structure is characteristic of Bach's fugues, which often feature:

* A subject (theme) that is introduced and vari...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about the major s... | Q: What is a counter-intuitive fact about the majo... | 0.23 |
| Who is most associated with the loudness war? | Q: Who is most associated with the loudness war? A... | 0.03 |
| What practical use does the British Invasion have ... | Q: What is one famous application of the British I... | 0.45 |
| How has the major scale changed over time? | Q: How has the major scale changed over time? A: T... | 0.02 |
| Give a concrete example of Bach's fugues. | Q: Give a concrete example of Bach's fugues. A: A ... | 0.00 |
| What field does the synthesizer belong to? | Q: What field does the synthesizer belong to? A: T... | 0.00 |
| What is a counter-intuitive fact about song writin... | Q: What is a counter-intuitive fact about song wri... | 0.00 |
| What is one famous application of verse-chorus for... | Q: What is one famous application of verse-chorus ... | 0.00 |
| Give a concrete example of the synthesizer. | Q: Give a concrete example of the synthesizer. A: ... | 0.00 |
| Who is most associated with the saxophone? | Q: Who is most associated with the saxophone? A: J... | 0.01 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.