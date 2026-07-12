# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 02:35:06
- Teacher: Ollama llama3.2:3b
- Domain: medicine
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 20263
- Cost: $0 (local Ollama)
- Time: 455.2s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about vaccines?

**Teacher answer**: One counter-intuitive fact about vaccines is that they can cause more harm than good in rare cases, such as vaccine-associated adverse events (VAEs). According to the Centers for Disease Control and Prevention (CDC), "the benefits of vaccination far outweigh the risks" (source: CDC, 2022). However, a study published in The Lancet found that the risk of serious adverse events from vaccines is estim...

### Q: Who is most associated with the menstrual cycle?

**Teacher answer**: Hippocrates (460-370 BCE) is often referred to as the "father of medicine" and is most associated with understanding the menstrual cycle. He described the female reproductive system and menstrual cycle in his writings, including the concept of a monthly flow of blood from the uterus.

Source: Hippocrates, "On Women", translated by W.H.S. Jones (1928)

### Q: What practical use does the digestive system have today?

**Teacher answer**: The digestive system has several practical uses today:

1. **Bioremediation**: Microorganisms in the gut can break down pollutants and toxins, helping to clean contaminated soil and water (source: "Microbial Bioremediation" by NASA).
2. **Protein production**: Yeast and bacteria in the gut can be used to produce proteins for food, pharmaceuticals, and other industries.
3. **Biofuel production**: M...

### Q: How has vaccines changed over time?

**Teacher answer**: Vaccines have undergone significant changes over time, driven by advances in technology, understanding of immunology, and global health needs.

1. **Early vaccines (1796-1930s)**: First vaccines were made from weakened or killed pathogens, such as Edward Jenner's smallpox vaccine (source: [1]). These early vaccines had limited efficacy and required multiple doses.
2. **Inactivated vaccines (1940s-...

### Q: Give a concrete example of Alzheimer's disease.

**Teacher answer**: A concrete example of Alzheimer's disease is the case of Terry Fox, a Canadian athlete who was diagnosed with early-onset Alzheimer's disease in 1998.

Source: "Terry Fox" by CBC News (2019)

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about vaccines? | Q: What is a counter-intuitive fact about vaccines... | 0.25 |
| Who is most associated with the menstrual cycle? | Q: Who is most associated with the menstrual cycle... | 0.01 |
| What practical use does the digestive system have ... | Q: What practical use does the digestive system ha... | 0.51 |
| How has vaccines changed over time? | Q: How has vaccines changed over time? A: Vaccines... | 0.52 |
| Give a concrete example of Alzheimer's disease. | Q: Give a concrete example of Alzheimer's disease.... | 0.50 |
| What field does asthma belong to? | Q: What field does asthma belong to? A: Asthma bel... | 0.01 |
| What is a counter-intuitive fact about anaphylaxis... | Q: What is a counter-intuitive fact about anaphyla... | 0.27 |
| What is one famous application of the lymphatic sy... | Q: What is one famous application of the lymphatic... | 0.00 |
| Give a concrete example of asthma. | Q: Give a concrete example of asthma. A: A 10-year... | 0.50 |
| Who is most associated with x-rays? | Q: Who is most associated with x-rays? A: Wilhelm ... | 0.02 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.