# How do Humans & LLMs Reason about Creativity?
Code and data to reproduce the results from our CogSci Paper "How do Humans and Language Models Reason About Creativity? A Comparative Analysis"

# Dependencies
A recent version of Python and R is needed to reproduce all results. We reccomend installing Python using Anaconda and setting up a virtual enviroment, and using R with RStudio.
## Python Dependencies
1. Numpy
2. Scipy
3. Pandas
4. tqdm
5. anthropic
6. matplotlib
7. seaborn
8. google generativeai
9. openai
10. nltk

There are some additional analyses using topic models that require extra packages, we did not report results on these attempts in the paper so running them is not necessary for reproducibility.

## R Dependencies
1. readr
2. cocor
3. dplyr
4. tidyr

# Reproducing our Results
The `scripts` folder contains all analysis notebooks for reproucing results. The bulk of this is in `CogSciHumanStudy.ipynb`, which has code for correlations, significance tests, linguistic analysis, and graphs. `RAnalysis.R` is specifically for checking significant differences in correlations across conditions. `dpt_few-shot_originality.py` runs an LLM experiment to complete the creativity task described in the paper.

The `data` folder contains human and LLM responses used to report these results, along with linguistic analysis using AI. Using this data, it is possible to reproduce our results without needing to make API calls, but the code also supports this option.

# Citation
If you use this work, please cite us:
```
@article{laverghetta2025humans,
  title={How do Humans and Language Models Reason About Creativity? A Comparative Analysis},
  author={Laverghetta Jr, Antonio and Chakrabarty, Tuhin and Hope, Tom and Pronchick, Jimmy and Bhawsar, Krupa and Beaty, Roger E},
  journal={arXiv preprint arXiv:2502.03253},
  year={2025}
}
```
 
