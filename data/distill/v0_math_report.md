# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-24 22:05:49
- Teacher: Ollama llama3.2:3b
- Domain: math
- Curriculum size: 50
- Examples collected: 50
- Total tokens (in+out): 10782
- Cost: $0 (local Ollama)
- Time: 427.0s

## Example queries and teacher answers

### Q: What problem does the quadratic formula solve?

**Teacher answer**: The quadratic formula solves for the roots (solutions) of a quadratic equation in the form ax² + bx + c = 0.

The quadratic formula is:

x = (-b ± √(b² - 4ac)) / 2a

This formula finds the values of x that make the equation true, given coefficients a, b, and c.

### Q: How does the law of large numbers relate to similar concepts?

**Teacher answer**: The Law of Large Numbers (LLN) is a fundamental concept in probability theory that states that as the number of trials or observations increases, the observed frequency of an event will converge to its theoretical probability.

Similar concepts related to LLN include:

1. **Law of Averages**: This concept is often used interchangeably with LLN, emphasizing that extreme events are rare and will eve...

### Q: Compare the fundamental theorem of calculus with one alternative approach.

**Teacher answer**: The Fundamental Theorem of Calculus (FTC) states that differentiation and integration are inverse processes. An alternative approach is the Riemann-Stieltjes integral, which also relates differentiation and integration.

In the FTC, we have:

∫f'(x)dx = f(x) + C

where f(x) is a function and C is a constant.

The Riemann-Stieltjes integral, on the other hand, uses a different type of "increment" (...

### Q: What is the simplest possible description of the quadratic formula?

**Teacher answer**: The quadratic formula can be described as:

x = (-b ± √(b² - 4ac)) / 2a

Source: "Algebra and Trigonometry" by James Stewart, 9th edition.

### Q: Explain linear algebra in 3 sentences.

**Teacher answer**: Linear algebra is a branch of mathematics that deals with the study of linear equations, vector spaces, and linear transformations. It provides a framework for solving systems of equations, finding eigenvalues and eigenvectors, and performing operations on vectors and matrices. The core concepts include vector addition, scalar multiplication, matrix multiplication, and determinants, which are used...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What problem does the quadratic formula solve? | Q: What problem does the quadratic formula solve? ... | 0.29 |
| How does the law of large numbers relate to simila... | Q: How does the law of large numbers relate to sim... | 0.00 |
| Compare the fundamental theorem of calculus with o... | Q: Compare the fundamental theorem of calculus wit... | 0.02 |
| What is the simplest possible description of the q... | Q: What is the simplest possible description of th... | 0.00 |
| Explain linear algebra in 3 sentences. | Q: What is the simplest possible description of li... | 0.03 |
| What is the historical origin of Bayes theorem? | Q: What is the historical origin of Bayes theorem?... | 0.49 |
| What problem does the central limit theorem solve? | Q: What problem does the central limit theorem sol... | 0.00 |
| What problem does the central limit theorem solve? | Q: What problem does the central limit theorem sol... | 0.00 |
| What is Bayes theorem used for? | Q: How does Bayes theorem relate to similar concep... | 0.00 |
| How does matrix multiplication relate to similar c... | Q: How does matrix multiplication relate to simila... | 0.00 |

## Takeaways

- 50 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.