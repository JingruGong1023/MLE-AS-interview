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

   L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. 

4. **Will Linear regression with L1 regularization be affected by multi-linearity?**

   No

5. **What is L1 Regression, what's the difference with L2**

   instead of adding squared w, we add absolute value of w in the equation

6. **How does L2 optimize model**

   The idea here is that we're decreasing the "slope" along each of the feature **by pushing the coefficients towards zero**. -> minimize weights, this prevents model from overfitting

7. **If we want to reduce number of features , which one to choose, L1,L2 and Why?**

   Lasso(L1) 

   L1 tends to shrink coefficients to zero, whereas L2 tends to shrink coefficients evenly. So when we want to drop features or do feature selection, we use **L1 instead of L2**

8. **What is ElasticNet, and how is it related to Lasso and Ridge?**

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

    LR is a supervised machine learning algorithm that can be used to model the probability of a certain data points belong to certain class or event , and usually used for binary classification, also the data should be linear separable.

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

    a supervised , non-parametric method that calculates y_hat using the average value or most common class of its k-nearest points

    dimension reduction should be applied before knn

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

​		KNN represents a supervised classification algorithm that will give new data points accordingly to the k number or the closest data points, while k-means clustering is an unsupervised clustering algorithm that gathers and groups data into k number of clusters.

20. 

</br>

## Trees

1. **How the tree will be split in decision trees ?**

2. **Why do non-tree models, such as Adaboost, SVM, LR, KNN, KMeans need standarization?**

   For linear model, when feature values varies too much, the gradient descent is hard to converge

3. **Why trees don't need standarization**

​		because scale of data won't affect how the trees be splited, in another word, it won't affect the ranking

3. **What's the problem with DT?**

   Overfitting

4. **Is DT linear or nonlinear**

   non linear

5. **What is ensemble Learning and what types are there?**

   · `Bagging` involves fitting many decision trees on different samples of the same dataset and averaging the predictions.

   · `Stacking` involves fitting many different models types on the same data and using another model to learn how to best combine the predictions

   · `Boosting` is an iterative strategy for adjusting an observations’ weight based on their previous classification-> **building strong learner from weaker learners**

6. **What is bootstrap, and how it works**

   sampling with replacement

7. **Which versions of trees can deal with both categorical variable and missing? which versions can only deal with missing**

   1. Lightgbm, catboost
   2. xgBoost

8. **What is ID3, what's the problem with it**

   information gain, the smaller h, the more pure for X. ID3 prefer feature with more levels, such as ID. no matter which id is chosen, the leaf will have high purity, -> overfitting

9. **What is Gain Ratio, how does it solve the problem ID3 caused?**

   $Gain\ ration = information \ gain/entropy$

   $information \ gain = entropy \ before \ splitting \ - \  entropy \ after \ splitting $

10. **What is Gini index, how does it work?**

    find the **best feature** and the **best threshold** to **minimize impurity**

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

24. **How does random forest acieves model independence**

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

    All boosting models iteratively try to improve a model built up from **weak learners**.

30. **What is Gradient Boosting , and how it works**

31. **Compare Gradient Boosting with Adaboosting**

32. **What are the advantages and disadvantages of GBDT**

    slow to train but fast to predict,more accurate than RF

33. **How to tune GBDT**

34. **What are the differences and similarities between Random Forest and GBDT**

    GBDT: faster in prediction, slower in training, more accurate , shallower trees, smaller model size, more sensitive to outliers

    GBDT is calculated by combining and learning from multiple trees, RF is by most votes

35. **Why is GBDT is faster in prediction. How to deal with slow training in GBDT**

    because **prediction can happen in parallel**, over all the trees + each tree in GBDT is usually much shallower than each tree in RF, and thus faster in traversal

    we can use XGBoost to optimize

36. **What is XGBoost**

    a decision-tree-based ensemble Machine Learning algorithm that uses a [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework

37. **When do you want to use Trees instead of Linear Models**

    non linear model

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

    only xgboost need feature encoding

32. **How do CatBoost, LightGBM and xgBoost deal with missing values**

    **CatBoost**: three mode: forbidden, min, max (does not need to deal with categorical feature)

    **Lightgbm**: uses NA (NaN) to represent missing values by default

    **xgBoost**:  it creates a third branch as well for missing values and will automatically learn which direction to go when a value is missing



</br>

## Model Evaluation

1. **What are the common metrics for Binary Classification**

   precision, recall, auc

2. **What does Precision and recall mean? why do we need F1 score?**

   **precision** = of all the predicted positive samples, how many of them are truly positive

   **Recall** = of all the real positive samples, how many of them our model predicted to be positive. = TPR

   **F1 Score** is the trade off between precision and recall

3. **What is Precision-Recall curve, which point is the best**

   Trade-off between precision and recall, recall on the x axis, precision on the y axis. The best point is at the top right

   it considers one threshold at a time

4. **When should we use precision, recall ?**

   precision helps when the false positive cost is high (movie recommendation)

   recall helps when the false negative cost is high (cancer detection)

5. **What is ROC, AUC, which point is the best**

   False positive rate on the x axis, True positive rate on the y axis

   the best point is on the top left

   - AUC is a ranking metric, it takes all possible thresholds into account, which means that it's **independent of the default thresholds**

6. **Threshold based method**

   Accuracy, precision, recall, f1

7. **What are the common metrics for multi-class classification**

   confusion matrix

8. **Averaging strategies in metrics for Multi-class classification**

   Macro, micro, weighted

9. **Ranking based method**

   ROC AUC

10. **What are the common metrics for regression**

    MSE, MAE , SSE, RMSE

11. **What is imbalanced data, and what  are the two sources of them**

12. **Two basic approaches in dealing with imbalanced data**

13. **what is SMOTE**

14. **How cross-validation works**

15. **What is bias , what does it mean to have high bias**

    the amount that a model's prediction differs from the target value

    high bias-> underfitting -> bad performance in training data

16. **What is variance ,  what does it mean to have high variance**

    how spread out the data is

    high variance -> overfitting -> bad at generalization 

17. **What is bayesian error**

    bayes error rate *is the lowest possible error rate for any classifier of a random outcome* and is analogous to the **irreducible error**

18. **what is the total error of a model**

    $bias^2+\ Variance \ + \ irreducible \ error$

19. **How to trade off variance and bias**

    in all cases, variance decreases, bias increases, we need to find the trade-off to minimize the error

20. **How to calculate the feature importance separately in Regressions and CART**

    CART: ranks variables by their ability to minimize impurity, averages across all trees

    

</br>

## Clustering

1. **What is Kmeans, and how it works**

   We want to add k new points to the data we have, each one of those points is call centroid, will be going around and trying to find a center in the middle of the k clusters we have. Algorithm stops when the points stop moving

   Hyperparameter: k – number of clusters we want

   Steps:

   Assign initial values to each cluster 

   Repeat: assign each point from training to each cluster that is closest to it in mean, recalculate the mean of each cluster

   If all clusters’ mean stop changing, algorithm stops

   K-mode used for categorical data

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

   Goal: minimize info loss, increase the interpretability in lower dimension, preserve data structure

2. **What do we need PCA and what does it do? **

   Project data onto orthogonal vectors that maximize variance

   PCA tries to find a lower dimensional surface project , such that the sum of the squared projection error is minimized

3. **What is the difference between logistic regression and PCA?**

4. **What is the drawback in PCA**

    Outliers

5. **how do we know the dimensions of the data with principle components**

    k- dimensional = k principle component

6. **How are the principle components ranked **

    pc are ranked by the proportion of variance explained by $ \frac{\lambda_i}{\sum \lambda} $

7. **What is the eigenvector in PCA**

    in PCA, the eigenvectors are uncorrelated and represents principle components

8. **What are the two pre-processing steps that should be applied before doing PCA?**

    mean normalization and feature scaling

9. **If we want only the 90% varaince, how to choose PC**

10. **PCA calculation details**

    https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues

11. **What if after pca, we still left with 300 dimensions**

     two reasons:

     1. The pca does not work:

        that means your features either have non-linear relationships or no relationships at all. go ahead choosing other kernel methods, such as t-SNE

        or you forgot to normalize your data..

     2. pca works, any reduction is good, we can't remove too much dimensions because we will lose lots of information. 

12. **Other than PCA, what else methods do you know for dimensionality reduction**

     T-SNE

13. **What is T- SNE**

     Focus on keeping very similar data close 

     

14. **How does T-SNE preserve local structure** 

     **using student t-distribution** to compute the similarity between two points in lower-dimensional space.

15. **Why does t-sne uses t distribution to compute the similarity in lower dimensional space**

     because t-distribution creates the probability distribution of points in lowers dimensional space, reduce the crowd issue

16. 

17. 

18. 

19. 

20. 

</br>







