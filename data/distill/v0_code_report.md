# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-24 22:05:41
- Teacher: Ollama llama3.2:3b
- Domain: code
- Curriculum size: 50
- Examples collected: 50
- Total tokens (in+out): 11184
- Cost: $0 (local Ollama)
- Time: 427.6s

## Example queries and teacher answers

### Q: What problem does binary search solve?

**Teacher answer**: Binary search solves the problem of finding an element in a sorted list or array efficiently.

Given a sorted list of n elements, where n is even, and a target value to be searched, binary search finds the index of the target value in O(log n) time complexity. This makes it much faster than linear search (O(n)) for large datasets.

Source: Wikipedia - Binary Search

### Q: How does object-oriented programming relate to similar concepts?

**Teacher answer**: Object-Oriented Programming (OOP) shares similarities with other design paradigms and concepts:

1. **Modular Programming**: OOP breaks down code into self-contained modules (objects), reducing coupling between them, much like modular programming.
2. **Functional Programming**: Both OOP and functional programming aim to abstract away low-level details, promoting modularity and reusability.
3. **Ev...

### Q: Compare dynamic programming with one alternative approach.

**Teacher answer**: Dynamic Programming (DP) and Memoization are two alternative approaches that can be used to solve complex problems efficiently.

**Similarities:** Both DP and Memoization aim to avoid redundant calculations by storing intermediate results, reducing computational complexity.

**Key differences:**

* **Dynamic Programming**: A top-down approach where the problem is broken down into smaller sub-probl...

### Q: What is the simplest possible description of binary search?

**Teacher answer**: Binary search is an algorithm that finds an element in a sorted list by repeatedly dividing the list in half and searching for the target element in one of the two halves.

(source: Wikipedia, "Binary Search")

### Q: Explain merge sort in 3 sentences.

**Teacher answer**: Merge sort is a divide-and-conquer algorithm that splits an unsorted list into two halves, recursively sorts each half, and then merges them back together in sorted order. The process repeats until only one element remains, resulting in a fully sorted list. This approach ensures stability and efficiency, with a time complexity of O(n log n) for large datasets.

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What problem does binary search solve? | Q: What problem does binary search solve? A: Binar... | 0.26 |
| How does object-oriented programming relate to sim... | Q: How does object-oriented programming relate to ... | 0.07 |
| Compare dynamic programming with one alternative a... | Q: Compare dynamic programming with one alternativ... | 0.05 |
| What is the simplest possible description of binar... | Q: What is the simplest possible description of bi... | 0.18 |
| Explain merge sort in 3 sentences. | Q: Explain merge sort in 3 sentences. A: Merge sor... | 0.01 |
| What is the historical origin of recursion? | Q: What are common misconceptions about recursion?... | 0.46 |
| What problem does functional programming solve? | Q: What problem does functional programming solve?... | 0.03 |
| What problem does functional programming solve? | Q: What problem does functional programming solve?... | 0.03 |
| What is recursion used for? | Q: What are common misconceptions about recursion?... | 0.28 |
| How does regular expressions relate to similar con... | Q: Compare regular expressions with one alternativ... | 0.02 |

## Takeaways

- 50 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.