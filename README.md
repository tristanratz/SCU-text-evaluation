# On the Role of Summary Content Units in Text Summarization Evaluation

Code for Paper: On the Role of Summary Content Units in Text Summarization Evaluation

At the heart of the Pyramid evaluation method
for text summarization lie human written sum-
mary content units (SCUs). These SCUs are
concise sentences that decompose a summary
into small facts. Such SCUs can be used to
judge the quality of a candidate summary, pos-
sibly partially automated via natural language
inference (NLI) systems. Interestingly, with
the aim to fully automate the Pyramid evalua-
tion, Zhang and Bansal (2021) show that SCUs
can be approximated from parsed semantic role
triplets (STUs). However, several questions cur-
rently lack answers, in particular i) Are there
other ways of approximating SCUs that can
offer advantages? ii) Under which conditions
are SCUs (or their approximations) offering
the most value? In this work, we examine two
novel strategies to approximate SCUs: gener-
ating SCU approximations from AMR mean-
ing representations (SMUs) and from large lan-
guage generation models (SGUs), respectively.
We find that while STUs and SMUs are compet-
itive, the best approximation quality is achieved
by SGUs. We also show through a simple
sentence-decomposition baseline (SSUs) that
SCUs (and their approximations) offer the most
value when ranking short summaries, but may
not help as much when ranking systems or
longer summaries

**Code will be released on 12.04.24**
