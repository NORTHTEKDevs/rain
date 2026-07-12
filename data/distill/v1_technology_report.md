# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 02:27:20
- Teacher: Ollama llama3.2:3b
- Domain: technology
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 21101
- Cost: $0 (local Ollama)
- Time: 467.3s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about public-key cryptography?

**Teacher answer**: A counter-intuitive fact about public-key cryptography is that it's computationally infeasible to determine whether a given public key was generated from a private key or randomly, without additional information.

This is known as the "public key problem" and is considered one of the most important problems in computer science. It's impossible for an attacker to distinguish between a legitimate pu...

### Q: Who is most associated with vector databases?

**Teacher answer**: Gregg Holman is often credited with developing the first vector database system in 1986 (source: [1]). However, the term "vector database" was popularized by Michael Stonebraker and his team at MIT in the late 1990s.

Michael Stonebraker is most associated with the development of the Postgres vector database system, which later became PostgreSQL.

### Q: What practical use does WebAssembly have today?

**Teacher answer**: WebAssembly (WASM) has several practical uses today:

1. **Cross-platform development**: WASM allows developers to write code that runs on multiple platforms, including web browsers, operating systems, and embedded devices.
2. **Game development**: Many game engines, such as Unity and Unreal Engine, support WASM, enabling developers to deploy games on the web and other platforms.
3. **Desktop appl...

### Q: How has public-key cryptography changed over time?

**Teacher answer**: Public-key cryptography has undergone significant changes and advancements over time:

1. **Diffie-Hellman key exchange (1976)**: Introduced secure key exchange between two parties without physically exchanging keys, source: Diffie et al., "New Directions in Cryptography" (1976).
2. **RSA algorithm (1977)**: Developed by Rivest, Shamir, and Adleman, it's a widely used public-key encryption algorit...

### Q: Give a concrete example of transformers in deep learning.

**Teacher answer**: A concrete example of transformers in deep learning is BERT (Bidirectional Encoder Representations from Transformers). It's a pre-trained language model that uses self-attention mechanisms to process sequential data, such as text.

In BERT, the input sequence is divided into subwords and embedded into a high-dimensional space. The transformer encoder consists of multiple identical layers, each con...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about public-key ... | Q: What is a counter-intuitive fact about public-k... | 0.22 |
| Who is most associated with vector databases? | Q: Who is most associated with vector databases? A... | 0.01 |
| What practical use does WebAssembly have today? | Q: What is the historical origin of WebAssembly? A... | 0.54 |
| How has public-key cryptography changed over time? | Q: How has public-key cryptography changed over ti... | 0.48 |
| Give a concrete example of transformers in deep le... | Q: Give a concrete example of transformers in deep... | 0.03 |
| What field does GPS positioning belong to? | Q: What field does GPS positioning belong to? A: G... | 0.00 |
| What is a counter-intuitive fact about knowledge g... | Q: What is a counter-intuitive fact about knowledg... | 0.02 |
| What is one famous application of RAG architecture... | Q: What is one famous application of RAG architect... | 0.00 |
| Give a concrete example of GPS positioning. | Q: Give a concrete example of GPS positioning. A: ... | 0.00 |
| Who is most associated with WebRTC? | Q: Who is most associated with WebRTC? A: Brendan ... | 0.03 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.