# COMP9418-Advanced-Topics-in-Statistical-Machine-Learning
This repository includes the projects of comp9418

## Assignment1
### Background
Breast cancer is the most common form of cancer and the second leading cause of cancer death in women.

Although it is not possible to say what exactly causes breast cancer, some factors may increase or change the risk for the development of cancer.

In this assignment, we try to use differenr kinds of Bayesian NetWorks for diagnosis of breast cancer.

### Task 1
In this part of assignment, we build a DAG to represent the relationship between different factors.

### Task 2
Estimating Bayesian Network parameters is the second task. The file bc.csv has 20,000 complete instances, i.e., without missing values.

### Task 3
This particular Bayesian Network has a variable that plays a central role in the analysis. The variable BC (Breast Cancer) can assume the values No, Invasive and InSitu. Accurately identifying its correct value
would lead to an automatic system that could help in early breast cancer diagnosis.

First use 10-fold cross-validation to split the dataset into training and test data.  Use the function learn_bayes_net(G, data, outcomeSpace) to learn
the Bayesian network parameters from the training set.

### Task 4
Naïve Bayes Classification

### Task 5
Tree-augmented Naïve Bayes Classification
Calculate the whole Bayesian Network is pretty expensive. If we can figure out which two factors are closely related, it will reduce a lot of calculation.
