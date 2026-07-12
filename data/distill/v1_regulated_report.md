# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 00:58:35
- Teacher: Ollama llama3.2:3b
- Domain: regulated
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 20183
- Cost: $0 (local Ollama)
- Time: 519.5s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about OSHA fall protection?

**Teacher answer**: A counter-intuitive fact about OSHA fall protection is that it's often more effective to use personal fall arrest systems (PFAS) with a harness and lanyard, rather than traditional guardrails or netting. This may seem counterintuitive because guardrails are often seen as the primary means of preventing falls, but PFAS can actually be more effective in certain situations, such as when working at he...

### Q: Who is most associated with DOT hazmat?

**Teacher answer**: The term "DOT" refers to the Department of Transportation (DOT). The individual most associated with DOT hazmat is likely James Ruckelaiher, but it's more commonly known that the person who created the original guidelines for hazardous materials transportation in the US is:

James Ruckelaiher

### Q: What practical use does ITAR export controls have today?

**Teacher answer**: ITAR (International Traffic in Arms Regulations) export controls primarily regulate the transfer of defense-related articles and services, including:

1. Military hardware (e.g., firearms, tanks, aircraft)
2. Technical data related to these items
3. Defense-related services (e.g., maintenance, repair)

These controls aim to prevent the proliferation of sensitive military technologies, particularly...

### Q: How has OSHA fall protection changed over time?

**Teacher answer**: OSHA's fall protection standards have undergone significant changes since their inception.

1. **1971**: The first Fall Protection Standard (29 CFR 1926.501) was introduced, requiring employers to provide a safe means of fall protection for employees working at heights.
2. **1989**: OSHA revised the standard to require personal fall arrest systems and lanyards (29 CFR 1926.502).
3. **1992**: The F...

### Q: Give a concrete example of EPA clean water act.

**Teacher answer**: The Clean Water Act (CWA) is a landmark environmental law in the United States that regulates pollution from point sources, such as factories and sewage treatment plants.

Example:

In 2010, the EPA issued a rule requiring coal-fired power plants to install new technology to reduce mercury emissions into waterways. This was done under Section 509 of the CWA, which allows the EPA to regulate pollut...

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about OSHA fall p... | Q: What is a counter-intuitive fact about OSHA fal... | 0.25 |
| Who is most associated with DOT hazmat? | Q: Who is most associated with DOT hazmat? A: The ... | 0.00 |
| What practical use does ITAR export controls have ... | Q: What practical use does ITAR export controls ha... | 0.25 |
| How has OSHA fall protection changed over time? | Q: How has OSHA fall protection changed over time?... | 0.37 |
| Give a concrete example of EPA clean water act. | Q: Give a concrete example of EPA clean water act.... | 0.22 |
| What field does MIFID II belong to? | Q: What field does MIFID II belong to? A: MIFID II... | 0.27 |
| What is a counter-intuitive fact about the Wassena... | Q: What is a counter-intuitive fact about the Wass... | 0.21 |
| What is one famous application of CCPA California ... | Q: Compare CCPA California privacy with one altern... | 0.26 |
| Give a concrete example of MIFID II. | Q: Give a concrete example of MIFID II. A: MIFID I... | 0.18 |
| Who is most associated with SOC 2 compliance? | Q: Who is most associated with SOC 2 compliance? A... | 0.00 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.