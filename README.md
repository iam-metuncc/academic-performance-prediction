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
    hypothesis_analysis.ipynb        [Old] Tests for RQ1 to RQ5 on the combined dataset using a nested cross-validation scheme
    flattened_cv.ipynb               The updated single-split hypothesis analysis presented in the paper.
    quick_statistical_results.ipynb  Paired t-tests between feature sets
    figures/                         Figures used in the thesis
    confusion_matrices/              Out-of-fold confusion matrices (PNG and PDF)
tests/                      pytest suite
    test_preprocessing.py   Preprocessing and validation tests
    test_data_loading.py    DatasetLoader tests
```

## Abstract

In many higher education institutions, the capstone poses a challenge for students. Identifying which students are likely to excel, and supporting the others, is thus beneficial for both students and instructors.
Academic performance prediction has been applied across a range of educational contexts. However, its use in capstone projects presents a distinct setting that warrants dedicated study, as it bridges academia and industry.
This industry-facing nature makes both technical competency and teamwork critical to success.
Assessing teamwork quality through a lightweight survey requires no dedicated data-collection infrastructure and is well-suited for periodic repetition as needed.
To the best of our knowledge, no existing work combines academic performance features with Teamwork Quality (TWQ) features in a capstone context, leaving this avenue of prediction unexplored.
This paper evaluates six machine learning models and five feature sets for classifying top performers in software engineering capstone courses, combining prior academic history with teamwork quality metrics.
Using data from 100 students in 26 teams across two academic years, we find that academic history predicts capstone outcomes above a majority-class baseline in both courses, most strongly in the design-oriented course, where all six models reached 74--87\% accuracy. Combining both feature sets reaches 78\% accuracy in the implementation-oriented course, which was 69\% with only TWQ features, and 72\% with only academic features.
Furthermore, prerequisite course grades alone retain comparable predictive power to the full academic history in the implementation course, suggesting that foundational knowledge is a key component of capstone project success.
