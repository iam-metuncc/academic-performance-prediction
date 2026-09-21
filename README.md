# Predicting Academic Performance in Software Engineering Capstone Projects Using Academic and Teamwork Features

M.Sc. thesis code, Computer Engineering, Middle East Technical University Northern Cyprus Campus (METU NCC).

This repository contains the data preprocessing, machine learning experiments, statistical analysis, and tests for a thesis that predicts student outcomes in the CNG 491 / CNG 492 software engineering capstone courses from prior academic history and Teamwork Quality (TWQ) survey scores.

## File structure

```
adamthesis/                 Python package with the experiment code
    data_labels.py          Column definitions: TWQ sections, course columns, prerequisite lists
    preprocessing.py        Cleaning and encoding of raw survey/transcript data, TWQ scoring, group-level aggregation
    data_loading.py         DatasetLoader: selects batch, feature set, and target; combines batches
    gridsearch.py           Model definitions, nested cross-validation, statistical tests, confusion matrices, result logging
notebooks/                  Analysis notebooks
    descriptive_statistics.ipynb     Side-by-side statistics for both cohorts
    Inter-batch_comparison.ipynb     Comparison of features across the two cohorts
    hypothesis_analysis.ipynb        Tests for RQ1 to RQ5 on the combined dataset
    quick_statistical_results.ipynb  Paired t-tests between feature sets
    figures/                         Figures used in the thesis
    confusion_matrices/              Out-of-fold confusion matrices (PNG and PDF)
tests/                      pytest suite
    test_preprocessing.py   Preprocessing and validation tests
    test_data_loading.py    DatasetLoader tests
```

## Abstract

The capstone project is the culminating requirement of most undergraduate Software Engineering programs. 
It poses unique challenges for students due to its team-based nature, long duration, and deliverable-based assessment.
Predicting a student's performance in the capstone project is thus valuable to both academic institutions and students.
Although academic performance prediction has been widely studied, its use in capstone projects presents a distinct setting that warrants dedicated study, as it bridges academia and industry.
This industry-facing nature makes both technical competency and teamwork critical to success.
To the best of our knowledge, no existing work combines academic performance features with Teamwork Quality (TWQ) features in a capstone context, leaving this avenue of prediction unexplored.
This thesis evaluates the performance of different machine learning models and various feature sets for predicting students' academic performance in software engineering capstone projects by combining prior academic history with teamwork quality metrics, sourcing the academic history from readily available academic transcripts and restricting teamwork quality metrics to a single lightweight survey.
Experiments conducted on students' data across two consecutive academic years demonstrate that both feature categories predict capstone outcomes above a random baseline, and that their combination achieves 78\% accuracy using a k-Nearest Neighbours classifier, outperforming either feature set alone.
Finally, we demonstrate that prerequisite course grades alone retain comparable predictive power to the full academic history, suggesting that foundational knowledge is a key component of capstone project success.
