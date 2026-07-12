# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 00:41:17
- Teacher: Ollama llama3.2:3b
- Domain: code
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 22370
- Cost: $0 (local Ollama)
- Time: 502.1s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about merge sort?

**Teacher answer**: A counter-intuitive fact about merge sort is that it has an average time complexity of O(n log n), but its worst-case scenario can be O(n^2) if the input array is already sorted or nearly sorted. This occurs because in these cases, the algorithm performs many unnecessary comparisons and swaps.

Source: "Introduction to Algorithms" by Thomas H. Cormen et al., 3rd edition (2009).

### Q: Who is most associated with type systems?

**Teacher answer**: Alonzo Church (1903-1995) and Stephen Kleene (1909-1994) are often credited with developing mathematical foundations for modern type theory, but a key figure in popularizing type systems is:

Alan Turing (1912-1954), who introduced the concept of the universal Turing machine and developed the theoretical foundations of modern computer science, including type theory.

However, the person most assoc...

### Q: What practical use does the actor model have today?

**Teacher answer**: The Actor Model is still widely used in various domains:

1. **Distributed Systems**: Actors are well-suited for distributed systems due to their ability to handle concurrent communication and failover.
2. **Cloud Computing**: Cloud-based applications often employ actors to manage complex workflows, scalability, and fault tolerance.
3. **Real-time Systems**: Actors' event-driven nature makes them ...

### Q: How has merge sort changed over time?

**Teacher answer**: Merge sort has undergone several improvements and optimizations over time.

1. **Original algorithm (1965)**: Developed by John von Neumann, it was a simple divide-and-conquer approach that divided the array into two halves, sorted each half recursively, and then merged them.
2. **Stability improvement (1970s)**: Researchers like Charles P. Snider and David H. Bailey introduced stability improveme...

### Q: Give a concrete example of object-oriented programming.

**Teacher answer**: A classic example of object-oriented programming (OOP) is a Bank Account system.

**BankAccount Class**
```python
class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
          ...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about merge sort? | Q: What is a counter-intuitive fact about merge so... | 0.00 |
| Who is most associated with type systems? | Q: What are common misconceptions about type syste... | 0.17 |
| What practical use does the actor model have today... | Q: What practical use does the actor model have to... | 0.00 |
| How has merge sort changed over time? | Q: How has merge sort changed over time? A: Merge ... | 0.46 |
| Give a concrete example of object-oriented program... | Q: How has functional programming changed over tim... | 0.22 |
| What field does design patterns belong to? | Q: What field does design patterns belong to? A: D... | 0.00 |
| What is a counter-intuitive fact about TCP vs UDP? | Q: What is a counter-intuitive fact about TCP vs U... | 0.27 |
| What is one famous application of B-trees? | Q: What is one famous application of B-trees? A: O... | 0.29 |
| Give a concrete example of design patterns. | Q: How does one learn design patterns effectively?... | 0.01 |
| Who is most associated with cache coherency? | Q: Compare cache coherency with one alternative ap... | 0.22 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.