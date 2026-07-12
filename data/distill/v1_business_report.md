# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 02:19:23
- Teacher: Ollama llama3.2:3b
- Domain: business
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 19234
- Cost: $0 (local Ollama)
- Time: 458.0s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about compound interest?

**Teacher answer**: A counter-intuitive fact about compound interest is that even small, consistent investments can grow significantly over time due to compounding. For example, if you invest $1,000 at an annual interest rate of 5% compounded annually for 20 years, the total amount would be approximately $1,368.79 (source: Investopedia). However, if you invest just $100 per year at the same interest rate and time fra...

### Q: Who is most associated with the customer lifetime value?

**Teacher answer**: Roger Bernhardt is often credited with coining the term "Customer Lifetime Value" (CLV), but it was Roger's colleague, Bill Bernhardt, who initially introduced the concept in 1960.

However, the modern understanding of CLV is largely attributed to Roger Bernhardt and his work on customer lifetime value models.

### Q: What practical use does Series A funding have today?

**Teacher answer**: Series A funding typically provides $2-30 million in capital to early-stage companies with a proven product-market fit, often in exchange for equity. This funding is used to:

1. Scale operations and expand teams (source: CB Insights)
2. Develop new products or features
3. Enhance marketing and sales efforts
4. Build strategic partnerships
5. Drive user acquisition and retention

Series A funding ...

### Q: How has compound interest changed over time?

**Teacher answer**: Compound interest has been around for thousands of years and its concept remains relatively consistent. However, the way it's applied and calculated has evolved.

In ancient Greece (around 500 BCE), the Greek mathematician Euclid described a method for calculating simple interest. The modern concept of compound interest emerged in the 17th century with the work of Dutch mathematician and physicist...

### Q: Give a concrete example of the Boston consulting matrix.

**Teacher answer**: The Boston Consulting Group (BCG) Matrix is a strategic planning tool used to evaluate and prioritize business units or products based on their market growth rate and relative market share.

Here's an example:

Let's say we have three products: Product A, Product B, and Product C.

| Product | Market Growth Rate | Relative Market Share |
| --- | --- | --- |
| A | 10% | 30% |
| B | 5% | 20% |
| C |...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about compound in... | Q: What is a counter-intuitive fact about compound... | 0.26 |
| Who is most associated with the customer lifetime ... | Q: Who is most associated with the customer lifeti... | 0.01 |
| What practical use does Series A funding have toda... | Q: What practical use does Series A funding have t... | 0.00 |
| How has compound interest changed over time? | Q: How has compound interest changed over time? A:... | 0.54 |
| Give a concrete example of the Boston consulting m... | Q: Give a concrete example of the Boston consultin... | 0.01 |
| What field does stock options belong to? | Q: What field does stock options belong to? A: Sto... | 0.00 |
| What is a counter-intuitive fact about cohort anal... | Q: What is a counter-intuitive fact about cohort a... | 0.22 |
| What is one famous application of MRR vs ARR? | Q: What are common misconceptions about MRR vs ARR... | 0.00 |
| Give a concrete example of stock options. | Q: Give a concrete example of stock options. A: Im... | 0.00 |
| Who is most associated with subscription economics... | Q: Who is most associated with subscription econom... | 0.03 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.