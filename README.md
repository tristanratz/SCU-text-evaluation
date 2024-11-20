# On the Role of Summary Content Units in Text Summarization Evaluation

Code for the paper: *On the Role of Summary Content Units in Text Summarization Evaluation*. The Paper is available at [arxiv](https://arxiv.org/abs/2404.01701) and [ACL anthology](https://aclanthology.org/2024.naacl-short.25/).

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

# Citation

If you find our work interesting, consider citing it:

```
@inproceedings{nawrath-etal-2024-role,
    title = "On the Role of Summary Content Units in Text Summarization Evaluation",
    author = "Nawrath, Marcel  and
      Nowak, Agnieszka  and
      Ratz, Tristan  and
      Walenta, Danilo  and
      Opitz, Juri  and
      Ribeiro, Leonardo  and
      Sedoc, Jo{\~a}o  and
      Deutsch, Daniel  and
      Mille, Simon  and
      Liu, Yixin  and
      Gehrmann, Sebastian  and
      Zhang, Lining  and
      Mahamood, Saad  and
      Clinciu, Miruna  and
      Chandu, Khyathi  and
      Hou, Yufang",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-short.25",
    doi = "10.18653/v1/2024.naacl-short.25",
    pages = "272--281",
    abstract = "At the heart of the Pyramid evaluation method for text summarization lie human written summary content units (SCUs). These SCUs areconcise sentences that decompose a summary into small facts. Such SCUs can be used to judge the quality of a candidate summary, possibly partially automated via natural language inference (NLI) systems. Interestingly, with the aim to fully automate the Pyramid evaluation, Zhang and Bansal (2021) show that SCUs can be approximated by automatically generated semantic role triplets (STUs). However, several questions currently lack answers, in particular: i) Are there other ways of approximating SCUs that can offer advantages?ii) Under which conditions are SCUs (or their approximations) offering the most value? In this work, we examine two novel strategiesto approximate SCUs: generating SCU approximations from AMR meaning representations (SMUs) and from large language models (SGUs), respectively. We find that while STUs and SMUs are competitive, the best approximation quality is achieved by SGUs. We also show through a simple sentence-decomposition baseline (SSUs) that SCUs (and their approximations) offer the most value when rankingshort summaries, but may not help as much when ranking systems or longer summaries.",
}
```

# Acknowledgements

This code is based on [atomicsents_amr](https://github.com/leoribeiro/atomicsents_amr) by Leonardo Ribeiro and Juri Opitz. Our original repository can be found under [this link](https://github.com/tristanratz/atomicsents_amr).
