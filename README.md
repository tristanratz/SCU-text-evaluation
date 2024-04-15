# On the Role of Summary Content Units in Text Summarization Evaluation

Code for Paper: On the Role of Summary Content Units in Text Summarization Evaluation \[[Link to Paper](https://arxiv.org/abs/2404.01701)]

At the heart of the Pyramid evaluation method for text summarization lie human written summary content units (SCUs). These SCUs are concise sentences that decompose a summary into small facts. Such SCUs can be used to judge the quality of a candidate summary, possibly partially automated via natural language inference (NLI) systems. Interestingly, with the aim to fully automate the Pyramid evaluation, Zhang and Bansal (2021) show that SCUs can be approximated from parsed semantic role triplets (STUs). However, several questions currently lack answers, in particular i) Are there other ways of approximating SCUs that can offer advantages? ii) Under which conditions are SCUs (or their approximations) offering the most value? In this work, we examine two novel strategies to approximate SCUs: generating SCU approximations from AMR meaning representations (SMUs) and from large language generation models (SGUs), respectively. We find that while STUs and SMUs are competitive, the best approximation quality is achieved by SGUs. We also show through a simple sentence-decomposition baseline (SSUs) that SCUs (and their approximations) offer the most value when ranking short summaries, but may not help as much when ranking systems or longer summaries.

# Generation

The generation of our samples for SGUs and SMUs can be found in the generation directory.

# Evaluation 

To evaluate the data run the script in the evaluation folder using 

```bash
pip install -R requirements.txt
cd evaluation
python3 extrinsic_evaluation.py
python3 intrinsic_evaluation.py
```

# Cite

```latex
@article{nawrath2024role,
      title={On the Role of Summary Content Units in Text Summarization Evaluation}, 
      author={Marcel Nawrath and Agnieszka Nowak and Tristan Ratz and Danilo C. Walenta and Juri Opitz and Leonardo F. R. Ribeiro and João Sedoc and Daniel Deutsch and Simon Mille and Yixin Liu and Lining Zhang and Sebastian Gehrmann and Saad Mahamood and Miruna Clinciu and Khyathi Chandu and Yufang Hou},
      year={2024},
      eprint={2404.01701},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Acknowledgements

This code is based on [atomicsents_amr](https://github.com/leoribeiro/atomicsents_amr) by Leonardo Ribeiro and Juri Opitz. Our original repository can be found under [this link](https://github.com/tristanratz/atomicsents_amr).
