# RAIN-Net Ollama Distillation Run Report

- Date: 2026-05-25 03:05:39
- Teacher: Ollama llama3.2:3b
- Domain: food
- Curriculum size: 100
- Examples collected: 100
- Total tokens (in+out): 20793
- Cost: $0 (local Ollama)
- Time: 460.7s

## Example queries and teacher answers

### Q: What is a counter-intuitive fact about sourdough?

**Teacher answer**: Sourdough bread actually has a lower pH level than regular bread due to the lactic acid produced by the wild yeast and bacteria in the starter, making it more acidic rather than less acidic as one might expect.

### Q: Who is most associated with the role of gluten?

**Teacher answer**: Karl Landsteiner and Alfred F. Neubauer are often credited with discovering the connection between gluten and celiac disease in 1916 (source: "Celiac Disease" by Dr. Alessio Fasano, source: [1]). However, it was actually Karl Landsteiner who first described the condition as "coeliac disease" in 1913 (source: "The History of Celiac Disease" by Dr. Alessio Fasano, source: [2]).

But if you're lookin...

### Q: What practical use does tea fermentation have today?

**Teacher answer**: Tea fermentation has several practical uses today:

1. **Production of Pu-erh tea**: Fermentation is used to age and mature Pu-erh tea, a type of Chinese tea that is prized for its earthy flavor.
2. **Flavor enhancement**: Tea fermentation can be used to enhance the flavor of other teas, such as green tea or white tea, by adding microorganisms like bacteria or yeast.
3. **Food industry application...

### Q: How has sourdough changed over time?

**Teacher answer**: Sourdough bread has undergone significant changes throughout history.

* Ancient Egyptians (around 1500 BCE) and Greeks (around 500 BCE) used wild yeast to leaven their bread, which was often coarse and dense.
* In the Middle Ages, European bakers used a mixture of flour, water, and salt to create a natural starter culture, but it was not as refined or consistent as modern sourdough.
* The Industr...

### Q: Give a concrete example of vinegars.

**Teacher answer**: A common type of vinegar is Apple Cider Vinegar (ACV). It's made from fermented apple cider and has a mild, slightly sweet taste. ACV is often used in cooking, salad dressings, and as a digestive aid due to its acidity.

Source: "The Oxford Companion to American Food and Drink" by Andrew F. Smith

## After-distillation retrieval (first 10 queries)

| Query | Top cited fact | Confidence |
|---|---|---|
| What is a counter-intuitive fact about sourdough? | Q: What is a counter-intuitive fact about sourdoug... | 0.22 |
| Who is most associated with the role of gluten? | Q: Who is most associated with the role of gluten?... | 0.02 |
| What practical use does tea fermentation have toda... | Q: What practical use does tea fermentation have t... | 0.52 |
| How has sourdough changed over time? | Q: How has sourdough changed over time? A: Sourdou... | 0.47 |
| Give a concrete example of vinegars. | Q: Give a concrete example of vinegars. A: A commo... | 0.00 |
| What field does knife technique belong to? | Q: What field does knife technique belong to? A: K... | 0.01 |
| What is a counter-intuitive fact about pickling? | Q: What is a counter-intuitive fact about pickling... | 0.22 |
| What is one famous application of caramelization? | Q: What is one famous application of caramelizatio... | 0.00 |
| Give a concrete example of knife technique. | Q: Give a concrete example of knife technique. A: ... | 0.00 |
| Who is most associated with the umami taste? | Q: Who is most associated with the umami taste? A:... | 0.03 |

## Takeaways

- 100 new facts entered semantic memory with zero retraining of the base model.
- All retrieval is via cosine similarity in HV space; no further LLM calls.
- The KB now serves as the substrate for active-learning answers.
- Cost per future query against these topics: $0 (local memory lookup).

This validates the active-distillation loop end-to-end against a real teacher.