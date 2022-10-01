## General Basics

1. **What types are there for machine learning?**

   1. supervised
   2. unsupervised
   3. reinforcement - Reinforcement Learning is less supervised which depends on the agent in determining the output.

   </br>

## Preprocessing

1. **What is data preprocessing ?**

   data cleaning, normalization, feature engineering etc

2. **What is Feature Scaling/Data Normalization?**

   rescale the data/features to put them into range [0,1]

3. **What are different types of feature scaling?**

​		Standarization, MinMaxScaler, RobustScaler, Normalizer

4. **What is Standarization?**

   mean removal and variance scaling

5. **What models need standarization/rescaling?**

   most linear

6. **What models don't need standarization/rescaling?**

   Trees, non linear

7. **What is data leakage and How do we correct it?**

   Scaling/ normalizing on both the training and validation set

   Correct: put scaling/normalization in the loop of cross validation, so everytime we scale a little differently

8. **Ways to Encode Categorical Features. Cons and Pros**

9. **What models must encode categorical features first, what models don't**

   Random forest, xgboost must have encoding

10. **What is Target-based encoding, what is the disadvantage?**

    we use a single variable to encode the response

    Regression-> average; binary -> fraction; multi-class-> fraction

    Cons: may lead to data leakage, use with regularization to solve

11. **What is Discretization**

    provides a way to partition continuous features into discrete values(groups)

12. **Should we always remove missing values?**

    No

13. **Why can't we just simply drop missing values?**

    missing values might be meaningful

14. **What is simple imputer, and what are the drawbacks?**

    impute with mean, mode, 

    - lost the class information a lot, may lead to worse performance

15. **How to impute categorical data with simple imputer**

    use mode, or add another category : unknown

16. **What is iterative imputer**

    - Iterate: retrain after filling in
    - you do the first pass and impute data using the mean.
    - then you try to predict the missing features using a regression model trained on the non-missing features
    - then you iterate this until stuff doesn't change anymore.

17. **What is KNN imputer**

    - Find k nearest neighbors that have non-missing values.
    - Fill in all missing values using the average of the neighbors

18. **How do you handle outliers in your dataset**

    not all outliers should be removed

    

19. **What is Isolation Forests and how it works?**

    similar as random forest, built based on decision trees. And It is built on the fact the outliers are few and different. We identify the outliers by choosing the samples ending up in shorter branches which indicate it was easier for the tree to separate them from other observations. 

</br>

## Linear Models

1. **What is Linear Regression? Pros and Cons**

2. **Why do we need Ridge(L2) Regression and what is it?**

   deal with multicolinearity

3. **What is L2 Regression assumption**

4. **Will Linear regression with L1 regularization be affected by multi-linearity?**

   No

5. **What is L1 Regression, what's the difference with L2**

6. **If we want to reduce number of features , which one to choose, L1,L2 and Why?**

   Lasso

7. **What is ElasticNet, and how is it related to Lasso and Ridge?**

​	L2 helps generalization, L1 helps makes some features' weights to zero

1. **What are the assumptions for linear regression?**

   - Linear relationship between independent and dependent variable
   - error terms constance varaince
   - error term follows normal distribution
   - no correlation among error terms

2. **Does these assumptions for linear regression also apply to Logistic regression?** No

3. **What does F stat test?**

   F-stat is used to test whether there is relationship between any of the predictors and response

4. **What is VIF , and what is its cutoff value?**

   Check Multilinearity, 10

5. **Model evaluation metrics for Linear Regression**

   - MSE/MAE, MAE more robust to outliers
   - R^2 /Adjusted R2

6. **Why use absolute/squared value in MAE and MSE?**

7. **Compare MAE, MSE, RMSE**

8. **What does R2 measures, and what does 0 and 1 mean**

9. **When can R2 be misleading?**

10. **why do we use adjusted R2? and it will only increase in which situationw**

11. **What is Logistic Regression, pro and cons?**

12. **Why not use MSE in Logistic?**

13. **MLE assumes data follow which distribution**

14. **Why log odds can predict probability?**

    because it used sigmoid function, basically squashed linear function $w^Tx$ between [0,1]

15. **Difference and similarity between logistic and linear regression**

16. **Difference between SVM and Logistic Regression**

    SVM tries to find the best margin that separates the classes and this reduce the risk of error on data, while logistic regression have different decision boundaries on different weights

    SVM works well on unstructured data such as text, images, but LR works better on identified independent variables.

    The risk of over fitting is less on SVM 

17. **What is KNN and how it's working?**

18. **Does KNN has any assumption for the data distribution?** No

19. **What is Manhattan distance, Euclidean distance and hamming distance**

    Manhattan distance: sum of abs(a-b)

    Euclidean distance : sqrt of sum (a-b)^2

    Hamming distance : count of the differences between two vectors

20. **How to choose k value in KNN**

​		k too large : simple model, underfitting, high bias

​		k too small : complex model, overfitting, high variance

​		usually use square root of n, or use cv to choose the optimal k

18. **Why dimension reduction is often applied prior to KNN?**

​		For high-dimensional data, information is lost through equidistant vectors

19. **What's the difference between K-means and KNN?**

​		KNN represents a supervised classification algorithm that will give new data points accordingly to the k number or the 			closest data points, while k-means clustering is an unsupervised clustering algorithm that gathers and groups data into k number of clusters.

20. **How to choose K value in KNN**

​		small k: complicate model, overfitting

​		Large k: simple model, underfitting

</br>

## Trees

1. **How the tree will be split in decision trees ?**

2. **Why do non-tree models, such as Adaboost, SVM, LR, KNN, KMeans need standarization?**

   For linear model, when feature values varies too much, the gradient descent is hard to converge

3. **Why trees don't need standarization**

​		because scale of data won't affect how the trees be splited, in another word, it won't affect the ranking

3. **What's the problem with DT?**

4. **Is DT linear or nonlinear**

5. **What is ensemble Learning and what types are there?**

6. **What is bootstrap, and how it works**

7. **Which versions of trees can deal with both categorical variable and missing? which versions can only deal with missing**

   1. Lightgbm, catboost
   2. xgBoost

8. **What is ID3, what's the problem with it**

   information gain, the smaller h, the more pure for X. ID3 prefer feature with more levels, such as ID. no matter which id is chosen, the leaf will have high purity, -> overfitting

9. **What is Gain Ratio, how does it solve the problem ID3 caused?**

   $Gain\ ration = information \ gain/entropy$

   $information \ gain = entropy \ before \ splitting \ - \  entropy \ after \ splitting $

10. **What is Gini index, how does it work?**

    gini index = 1- sum(p^2), p is the probability of each class

11. **How to calculate Entropy index/information gain**

    $Entropy = -\sum_j{p_j*log2(p_j)}$

12. **What are the impurity criterias for Regression tree and classification tree respectively**

13. **如果你有100个节点要做分类，怎么训练**

14. **what is the impurity index of entropy and impurity index of gini coefficients respectively, with classes with the same probability**

    Entropy = 1, gini = 0.5

15. **Increase which hyperparameters will make RF overfit**

    1. depth of trees
    2. Max number of features used

16. **Can we mix feature types in one tree? can we have the same features appear multiple times**

    yes, yes

17. **How to calculate the feature importance in tree?**

    Feature importance is calculated as **the decrease in node impurity weighted by the probability of reaching that node**.

18. **What is the difference between generalization and extrapolation**

    For `generalization`, usually you make this `IID` assumption that you draw **data from the same distribution**, and you have    some samples from the distribution from which you learn, and other samples from a distribution which you want to predict.

    For `extrapolation`, the distribution that I want to try to predict on was actually different from the distribution I learned on because they’re completely disjoint

19. **How can bootstrap improve model**

20. **What is Random Forest and how it works? Pro and Cons**

21. **How does RF introduce randomness?**
    A. by creating boostrap samples
    B. by selecting a random subset of features for each tree

22. **Will tree model be affected by multi-linearity?**

    no

23. **Which aspect of random forest uses bagging?**

    1. each tree is trained with random sample of the data
    2. each tree is trained with a subset of features

24. **How does random forest acieves model independend**

25. **What does proximity means in RF?**

    The term `proximity` means the closeness or nearness between pairs of cases

26. **How to randomize RF?**

    Row sampling & Column sampling: use bagging to make samples randomized

27. **How to tune Random Forest?**

    Max_features : which is the max number of features you want to look at each split -> too many feature might cause overfitting

    n_estimators: the number of trees you want to build before making decisions -> too few trees might cause overfitting

    Pre_prune : by setting max depth, max leaf_nodes, and min sample splits -> too deep might cause overfitting

28. **What is the diffference and similarity between Out-of-Bag score and CV score**

29. **What is Boosting, and what are the common algorithms using boosting**

30. **What is Gradient Boosting , and how it works**

31. **Compare Gradient Boosting with Adaboosting**

32. **What are the advantages and disadvantages of GBDT**

33. **How to tune GBDT**

34. **What are the differences and similarities between Random Forest and GBDT**

    GBDT: faster in prediction, slower in training, more accurate , shallower trees, smaller model size, more sensitive to outliers

    GBDT is calculated by combining and learning from multiple trees, RF is by most votes

35. **Why is GBDT is faster in prediction. How to deal with slow training in GBDT**

    because **prediction can happen in parallel**, over all the trees + each tree in GBDT is usually much shallower than each tree in RF, and thus faster in traversal

    we can use XGBoost to optimize

36. **What is XGBoost**

37. **When do you want to use Trees instead of Linear Models**

38. **How trees are pruned?**

https://medium.com/analytics-vidhya/post-pruning-and-pre-pruning-in-decision-tree-561f3df73e65

26. **Feature Importance**

27. **Why GBDT is more accurate then RF**

    随机森林是减少模型的方差(Variance)-> reduce overfitting，而GBDT是减少模型的偏差(Bias)-> reduce underfitting. That's why GBDT has high accuracy 

28. **When do you not want to use Tree based models**

    high dimensional data/unstructured data -> NN works better

29. **Tree Symmetry style for CatBoost, LightGBM and xgBoost**

    Catboost: symmetric

    LightGBM, xgBoost: asymmetric

30. **Splitting method for CatBoost, LightGBM and xgBoost**

31. **Do CatBoost, LightGBM and xgBoost need feature encoding?**

32. **How do CatBoost, LightGBM and xgBoost deal with missing values**



</br>

## Model Evaluation

1. **What are the common metrics for Binary Classification**

2. **What does Precision and recall mean? why do we need F1 score?**

3. **What is Precision-Recall curve, which point is the best**

4. **What is ROC, AUC, which point is the best**

5. **Threshold based method**

   Accuracy, precision, recall, f1

6. **What are the common metrics for multi-class classification**

7. **Averaging strategies in metrics for Multi-class classification**

   Macro, micro, weighted

8. **Ranking based method**

   ROC AUC

9. **What are the common metrics for regression**

10. **What is imbalanced data, and what  are the two sources of them**

11. **Two basic approaches in dealing with imbalanced data**

12. **what is SMOTE**

13. **How cross-validation works**

14. **What is bias , what does it mean to have high bias**

15. **What is variance ,  what does it mean to have high variance**

16. **What is bayesian error**

17. **what is the total error of a model**

18. **How to trade off variance and bias**

19. **How to calculate the feature importance separately in Regressions and CART**

</br>

## Clustering

1. **What is Kmeans, and how it works**

   We want to add k new points to the data we have, each one of those points is call centroid, will be going around and trying to find a center in the middle of the k clusters we have. Algorithm stops when the points stop moving

   Hyperparameter: k – number of clusters we want

   Steps:

   Assign initial values to each cluster 

   Repeat: assign each point from training to each cluster that is closest to it in mean, recalculate the mean of each cluster

   If all clusters’ mean stop changing, algorithm stops

2. **What are K-means' Pros and Cons**

   **pro**: 

   - easy to implement and tune with hyper parameter k. 
   - it guarantees convergence and easily adapt to new examples
   - low computational cost

   **cons**: 

   - Centroids can be dragged by outliers, or outliers might get their own cluster instead of being ignored. 
   - K-means algorithm can be performed in numerical data only.

3. **How to choose K, what metric**

   minimize variance within cluster, maximize variance with clusters

4. **What kind of clusters do you know**
5. **What is distortion function? Is it convex or non-convex?**
6. **Tell me about the convergence of the distortion function.**
7. **Topic: EM algorithm**
8. **What is the Gaussian Mixture Model?**
9. **Describe the EM algorithm intuitively.**
10. **What are the two steps of the EM algorithm**
11. **Compare GMM vs GDA.**

</br>

## Dimensionality Reduction

1. **Why do we need dimensionality reduction techniques?** 

   data compression, speeds up learning algorithm and visualizing data

2. **What do we need PCA and what does it do? **

   PCA tries to find a lower dimensional surface project , such that the sum of the squared projection error is minimized

3. **What is the difference between logistic regression and PCA?**

4. **What are the two pre-processing steps that should be applied before doing PCA?**

    mean normalization and feature scaling

5. **If we want only the 90% varaince, how to choose PC**

6. **PCA calculation details**

    https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues

7. **What if after pca, we still left with 300 dimensions**

    two reasons:

    1. The pca does not work:

       that means your features either have non-linear relationships or no relationships at all. go ahead choosing other kernel methods, such as t-SNE

       or you forgot to normalize your data..

    2. pca works, any reduction is good, we cann't remove too much dimensions because we will lose lots of information. 

8. **Other than PCA, what else methods do you know for dimensionality reduction**

    T-SNE

</br>







