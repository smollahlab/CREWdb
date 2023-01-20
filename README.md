# CREWdb : Optimizing Chromatin Readers, Writers and Erasers database using Machine Learning-based Approach.

## Abstract
Aberration in heterochromatin and euchromatin states contributes to various disease phenotypes. The transcriptional regulation between these two states is significantly governed by post-translational modifications made by three functional types of chromatin regulators: readers, writers, and erasers. Writers introduce a chemical modification to DNA and histone tails, readers bind the modification to histone tails using specialized domains, and erasers remove the modification introduced by writers. Altered regulation of these chromatin regulators plays a key role in complex diseases such as cancer, neurodevelopmental diseases, myocardial diseases, and embryonic development. Due to the reversible nature of chromatin modifications, we have the opportunity to develop therapeutic approaches targeting chromatin regulators. However, a limited number of chromatin regulators have been identified, and a subset of those identified have been ambiguously classified as multiple chromatin regulator types. Thus, we have applied machine learning-based approaches to predict and classify the functionality of chromatin regulator proteins, optimizing the accuracy of the first comprehensive database of chromatin regulators known as CREWdb.

## Contents
Machine Learning prediction and classification models used in CREWdb:
  1. K-Nearest Neighbors
  2. Decision Tree
  3. Random Forest
  4. Feed Forward Neural Network
  5. Support Vector Machine
  6. Logistic Regression
  7. Deep Neural Network
  8. Naive Bayes
  
Each of the models were trained using 5-fold Cross validation and Leave One Out Cross Validation (LOOCV). 

5-Fold-CREWdb.py : Machine learning pipeline for all mentioned models using 5-fold cross-validation.
LOOCV-CREWdb.py : Machine learning pipeline for all mentioned models using leave one out cross-validation.

## How to use
Requirements : 
  Python (v 3.9)
Packages:
  1. Numpy
  2. Pandas
  3. Scikit-learn
  4. matplotlib
  5. imbalanced-learn
  6. seaborn
  
 After setting up the Python environment with the above packages, the python scripts can be executed using the command: python filename.py
 Using Anaconda/ Miniconda is recommended to run the scripts and iPython notebooks.
 
 If you have any problems running our code, please feel free to contact us (smollah@wustl.edu, g.reetika@wustl.edu)
 
## Citation
If you find CREWdb useful in your research, please consider citing: 

Maya Natesan, Reetika Ghag, Mitchelle Kong, Min Shi, Shamim Mollah. “CREWdb: Optimizing Chromatin Readers, Erasers, and Writers Database using Machine Learning-Based Approach”. 2022. (doi.org/10.1101/2022.06.02.494594)
