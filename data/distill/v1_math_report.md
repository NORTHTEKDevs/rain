# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 00:49:55
- Teacher: Ollama llama3.2:3b
- Domain: math
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 20738
- Cost: $0 (local Ollama)
- Time: 516.8s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about linear algebra?

**Teacher answer**: One counter-intuitive fact in linear algebra is that the determinant of a matrix can be zero even if the matrix has no "zero rows" or "zero columns". This is known as the "determinant of a non-square matrix" and it's a fundamental property of determinants. 

Source: Linear Algebra and Its Applications by Gilbert Strang (source: Strang, G. (2016). Linear algebra and its applications. 5th ed.)

### Q: Who is most associated with the Fourier transform?

**Teacher answer**: The Fourier transform is most associated with Joseph Fourier (1768-1830), a French mathematician and physicist who first introduced the concept in the early 19th century.

### Q: What practical use does set theory have today?

**Teacher answer**: Set theory has numerous practical applications in various fields:

1. **Computer Science**: Set theory is used in data structures, algorithms, and database management to efficiently store and manipulate large datasets.
2. **Cryptography**: Set theory is essential for secure encryption methods, such as public-key cryptography, which relies on set operations like union, intersection, and difference....

### Q: How has linear algebra changed over time?

**Teacher answer**: Linear algebra has undergone significant changes and developments throughout its history.

**Early Beginnings:** The foundations of linear algebra date back to ancient civilizations, with contributions from mathematicians such as Diophantus (source: "A History of Mathematics" by Carl B. Boyer) in the 3rd century AD.

**17th-18th Centuries:** René Descartes introduced coordinates and matrix notatio...

### Q: Give a concrete example of the law of large numbers.

**Teacher answer**: A classic example is flipping a fair coin 100 times. According to the law of large numbers, as the number of trials (flips) increases, the observed frequency of heads or tails will converge to the expected probability of 0.5.

For instance, if we flip a coin 100 times and get 55 heads, it's likely that the next few flips will result in more heads than tails, but eventually, the number of heads and...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about linear alge... | Q: What is a counter-intuitive fact about linear a... | 0.21 |
| Who is most associated with the Fourier transform? | Q: Who is most associated with the Fourier transfo... | 0.21 |
| What practical use does set theory have today? | Q: What practical use does set theory have today? ... | 0.00 |
| How has linear algebra changed over time? | Q: What field does linear algebra belong to? A: Li... | 0.43 |
| Give a concrete example of the law of large number... | Q: Give a concrete example of the law of large num... | 0.27 |
| What field does group theory belong to? | Q: What field does group theory belong to? A: Grou... | 0.00 |
| What is a counter-intuitive fact about Fermat's la... | Q: What is a counter-intuitive fact about Fermat's... | 0.25 |
| What is one famous application of the prime number... | Q: What is one famous application of the prime num... | 0.27 |
| Give a concrete example of group theory. | Q: What field does group theory belong to? A: Grou... | 0.21 |
| Who is most associated with logic? | Q: Compare logic with one alternative approach. A:... | 0.28 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.