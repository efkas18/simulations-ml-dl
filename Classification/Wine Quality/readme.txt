								======== UCI MACHINE LEARNING REPOSITORY ========


Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009], http://www3.dsi.uminho.pt/pcortez/wine/).

Dataset Characteristics: Multivariate
Subject Area: Business
Associated Tasks: Classification, Regression
Feature Type: Real

=======================================================================================
* link: https://archive.ics.uci.edu/dataset/186/wine+quality
* Instances: 4898
* Features: 11
* Has Missing Values? : NO
=======================================================================================
Additional Information

For more information, read [Cortez et al., 2009].
Input variables (based on physicochemical tests):

   1 - fixed acidity

   2 - volatile acidity

   3 - citric acid

   4 - residual sugar

   5 - chlorides

   6 - free sulfur dioxide

   7 - total sulfur dioxide

   8 - density

   9 - pH

   10 - sulphates

   11 - alcohol

Output variable (based on sensory data): 

   12 - quality (score between 0 and 10)


=======================================================================================
======== Simulation Instructions ========

Regarding "Wine Quality" simulations, there are two approaches, machine learning and deep learning.
At both scenarios inside the code there is a specific variable "implementation", which separates the 
implementation user wants to execute.
If implementation is equal to '0', then will be executed "Binary Classification", by separating targets
to those are above 6.5 (good quality) and those are below (bad quality). In this case both ANN and Machine Learning models, success testing accuracy around 90%.
From the other hand, if implementation is not equal to '0', then the "Categorical Classification" will be selected,
which will present significant lower testing accuracy to 65%.







