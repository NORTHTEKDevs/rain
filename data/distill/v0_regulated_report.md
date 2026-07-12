# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-24 22:05:50
- Teacher: Ollama llama3.2:3b
- Domain: regulated
- Curriculum size: 50
- Examples collected: 50
- Total tokens (in+out): 9602
- Cost: $0 (local Ollama)
- Time: 424.8s

## Example queries and teacher answers

### Q: What problem does FAA part 91 solve?

**Teacher answer**: FAA Part 91 (Regulations Governing Operations of Civil Aircraft) solves the problem of regulating and standardizing private aircraft operations in the United States.

Specifically, it addresses issues such as:

* Ensuring safe operation of private aircraft
* Establishing rules for pilot certification, training, and experience requirements
* Regulating air traffic control procedures and communicati...

### Q: How does EPA clean water act relate to similar concepts?

**Teacher answer**: The Clean Water Act (CWA) is a federal law in the United States that regulates and enforces the protection of waters of the United States. It relates to similar concepts as follows:

* The Safe Drinking Water Act (SDWA): Both laws aim to protect public health by regulating water quality.
* The Resource Conservation and Recovery Act (RCRA): RCRA also deals with environmental protection, but focuses...

### Q: Compare GDPR consent with one alternative approach.

**Teacher answer**: GDPR consent is an opt-in requirement where users must explicitly agree to data processing for specific purposes. An alternative approach is "implied consent" or "opt-out", where users are not required to actively agree, but can choose to opt out of data processing if they wish.

Source: Article 4(11) GDPR

### Q: What is the simplest possible description of FAA part 91?

**Teacher answer**: FAA Part 91 is the general aviation rulemaking section that governs private aircraft operations in the United States.

Source: Federal Aviation Administration (FAA) - Title 14, Code of Federal Regulations (CFR), Part 91.

### Q: Explain OSHA fall protection in 3 sentences.

**Teacher answer**: OSHA fall protection regulations require employers to provide personal protective equipment (PPE) and training to employees working at heights above 6 feet. The use of guardrails, safety nets, or other fall protection systems is also required for certain types of work, such as construction or demolition projects. Employers must ensure that all necessary safety measures are in place before allowing...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What problem does FAA part 91 solve? | Q: What is the simplest possible description of FA... | 0.24 |
| How does EPA clean water act relate to similar con... | Q: How does EPA clean water act relate to similar ... | 0.26 |
| Compare GDPR consent with one alternative approach... | Q: Compare GDPR consent with one alternative appro... | 0.01 |
| What is the simplest possible description of FAA p... | Q: What is the simplest possible description of FA... | 0.25 |
| Explain OSHA fall protection in 3 sentences. | Q: Explain OSHA fall protection in 3 sentences. A:... | 0.48 |
| What is the historical origin of SEC rule 10b-5? | Q: What is the historical origin of SEC rule 10b-5... | 0.47 |
| What problem does IRS section 179 solve? | Q: What are common misconceptions about IRS sectio... | 0.03 |
| What problem does IRS section 179 solve? | Q: What are common misconceptions about IRS sectio... | 0.03 |
| What is SEC rule 10b-5 used for? | Q: What is the historical origin of SEC rule 10b-5... | 0.02 |
| How does ADA accessibility relate to similar conce... | Q: How does ADA accessibility relate to similar co... | 0.03 |

## Takeaways

- 50 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.