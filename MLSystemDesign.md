- [Resources](#resources)
  - [Rules of Machine Learning](#rules-of-machine-learning)
    - [Before ML](#before-ml)
    - [ML Phase I: Your First Pipeline](#ml-phase-i-your-first-pipeline)
      - [Monitoring](#monitoring)
      - [Your First Objective](#your-first-objective)
    - [ML Phase II: Feature Engineering](#ml-phase-ii-feature-engineering)
      - [Human Analysis of the System](#human-analysis-of-the-system)
      - [Training-Serving Skew](#training-serving-skew)
  - [ML Design Template](#ml-design-template)
    - [Overeview](#overeview)
    - [Data pipeline](#data-pipeline)
    - [Model Training](#model-training)
    - [Model Serving](#model-serving)
- [Grokking the Machine Learning Interview](#grokking-the-machine-learning-interview)
  - [Overview](#overview)
    - [Setting up a Machine Learning System](#setting-up-a-machine-learning-system)
  - [Practical ML Techniques/Concepts](#practical-ml-techniquesconcepts)
    - [Performance and Capacity Considerations](#performance-and-capacity-considerations)
      - [Complexities consideration for an ML system](#complexities-consideration-for-an-ml-system)
      - [Comparison of `training` and `evaluation` complexities](#comparison-of-training-and-evaluation-complexities)
      - [`Performance` and `Capacity` considerations in large scale system](#performance-and-capacity-considerations-in-large-scale-system)
      - [Layered/funnel based modeling approach](#layeredfunnel-based-modeling-approach)
    - [Training Data Collection Strategies](#training-data-collection-strategies)
      - [Collection techniques](#collection-techniques)
      - [Additional creative collection techniques](#additional-creative-collection-techniques)
      - [Train/Dev/Test splits](#traindevtest-splits)
      - [Quantity of training data](#quantity-of-training-data)
      - [Training data filtering](#training-data-filtering)
    - [Online Experimentation](#online-experimentation)
      - [Running an online experiment(A/B test)](#running-an-online-experimentab-test)
      - [Measuring results: Computing statistical significance](#measuring-results-computing-statistical-significance)
      - [Measuring long term effects](#measuring-long-term-effects)
    - [Embeddings](#embeddings)
      - [Text embeddings](#text-embeddings)
      - [Visual embedding](#visual-embedding)
      - [Learning embeddings for a particular learning task](#learning-embeddings-for-a-particular-learning-task)
      - [Network/Relationship-based embedding](#networkrelationship-based-embedding)
    - [Transfer Learning](#transfer-learning)
      - [Overview](#overview-1)
    - [Model Debugging and Testing](#model-debugging-and-testing)
      - [Building model v1](#building-model-v1)
      - [Deploying and debugging v1 model](#deploying-and-debugging-v1-model)
      - [Iterative model improvement](#iterative-model-improvement)
      - [Debugging large scale systems](#debugging-large-scale-systems)
  - [Search Ranking](#search-ranking)
    - [Problem Statement](#problem-statement)
      - [Clarifying questions](#clarifying-questions)
    - [Metrics](#metrics)
      - [Online metrics](#online-metrics)
      - [Offline metrics](#offline-metrics)
        - [NDCG](#ndcg)
    - [Architectural Components](#architectural-components)
      - [Query rewriting](#query-rewriting)
      - [Query understanding](#query-understanding)
      - [*Document selection](#document-selection)
      - [*Ranker](#ranker)
      - [Blender](#blender)
      - [Training data generation](#training-data-generation)
      - [Layerd model approach](#layerd-model-approach)
    - [Document Selection](#document-selection-1)
      - [Overview](#overview-2)
      - [Selection criteria](#selection-criteria)
      - [Relevance scoring scheme](#relevance-scoring-scheme)
    - [Feature Engineering](#feature-engineering)
      - [Searcher-specific features](#searcher-specific-features)
      - [Query-specific features](#query-specific-features)
      - [Document-specific features](#document-specific-features)
      - [Context-specific features](#context-specific-features)
      - [Searcher-document features](#searcher-document-features)
      - [Query-document features](#query-document-features)
    - [Training Data Generation](#training-data-generation-1)
      - [Data generation for pointwise approach](#data-generation-for-pointwise-approach)
        - [Positive and negative training examples](#positive-and-negative-training-examples)
      - [Data generation for pairwise approach](#data-generation-for-pairwise-approach)
        - [Human raters (offline method)](#human-raters-offline-method)
        - [User-engagement (online method)](#user-engagement-online-method)
    - [Ranking](#ranking)
      - [Overview](#overview-3)
      - [Stage 1 (Logistic regression)](#stage-1-logistic-regression)
      - [Stage 2 (LambdaMART, LambdaRank)](#stage-2-lambdamart-lambdarank)
    - [Filtering Results](#filtering-results)
      - [Result set after ranking](#result-set-after-ranking)
      - [ML problem](#ml-problem)
  - [Feed Based System](#feed-based-system)
    - [Problem statement](#problem-statement-1)
      - [Visualizing the problem](#visualizing-the-problem)
      - [Scale of the problem](#scale-of-the-problem)
    - [Metrics](#metrics-1)
      - [User actions](#user-actions)
      - [User engagement metrics](#user-engagement-metrics)
    - [Architecture](#architecture)
    - [Tweet selection](#tweet-selection)
      - [New Tweets](#new-tweets)
      - [New Tweets + unseen Tweets](#new-tweets--unseen-tweets)
      - [Network Tweets + interest / popularity-based Tweets](#network-tweets--interest--popularity-based-tweets)
    - [Feature Engnieering](#feature-engnieering)
      - [User-author features](#user-author-features)
      - [User-author similarity](#user-author-similarity)
      - [Author features](#author-features)
      - [User-Tweet features](#user-tweet-features)
      - [Tweet features](#tweet-features)
        - [Features based on Tweet’s content](#features-based-on-tweets-content)
        - [Features based on Tweet’s interaction](#features-based-on-tweets-interaction)
        - [Separate features for different engagements](#separate-features-for-different-engagements)
      - [Context-based features](#context-based-features)
      - [Sparse features](#sparse-features)
    - [Training Data Generation](#training-data-generation-2)
      - [Data generation through online user engagement](#data-generation-through-online-user-engagement)
      - [Balancing positive and negative training examples](#balancing-positive-and-negative-training-examples)
    - [Ranking](#ranking-1)
      - [Logistic regression](#logistic-regression)
      - [MART](#mart)
      - [Deep learning](#deep-learning)
        - [Multi-task neural networks](#multi-task-neural-networks)
      - [Stacking models and online learning](#stacking-models-and-online-learning)
    - [Diversity](#diversity)
      - [Diversity in Tweets’ authors](#diversity-in-tweets-authors)
      - [Diversity in tweets’ content](#diversity-in-tweets-content)
      - [Introducing the repetition penalty](#introducing-the-repetition-penalty)
    - [Online Experimentation](#online-experimentation-1)
  - [Recommendation System](#recommendation-system)
    - [Problem statement](#problem-statement-2)
      - [Scope of the problem](#scope-of-the-problem)
      - [Problem formulation](#problem-formulation)
    - [Metrics](#metrics-2)
      - [Online metrics](#online-metrics-1)
      - [Offline metrics](#offline-metrics-1)
        - [`mAP@N`](#mapn)
        - [`mAR@N`](#marn)
        - [`F1 score`](#f1-score)
        - [`RMSE`](#rmse)
    - [Architectural Components](#architectural-components-1)
      - [Candidate generation](#candidate-generation)
      - [Ranker](#ranker-1)
      - [Training data generation](#training-data-generation-3)
    - [Feature Engineering](#feature-engineering-1)
      - [User-based features](#user-based-features)
      - [Context-based features](#context-based-features-1)
      - [Media-based features](#media-based-features)
      - [Media-user cross features](#media-user-cross-features)
    - [Candidate Generation](#candidate-generation-1)
      - [Collaborative filtering](#collaborative-filtering)
        - [Nearest Neighborhood](#nearest-neighborhood)
        - [Matrix factorization](#matrix-factorization)
      - [Content-based filtering](#content-based-filtering)
      - [**Generate embedding using neural networks](#generate-embedding-using-neural-networks)
      - [Techniques’ strengths and weaknesses](#techniques-strengths-and-weaknesses)
    - [Training Data Generation](#training-data-generation-4)
      - [Generating training examples](#generating-training-examples)
      - [Balancing positive and negative training examples](#balancing-positive-and-negative-training-examples-1)
      - [Weighting training examples](#weighting-training-examples)
    - [Ranking](#ranking-2)
      - [Logistic regression or random forest](#logistic-regression-or-random-forest)
      - [Deep NN](#deep-nn)
      - [Re-ranking](#re-ranking)
  - [Ad Prediction System](#ad-prediction-system)
    - [Problem Statement](#problem-statement-3)
    - [Metrics](#metrics-3)
      - [Offline metrics](#offline-metrics-2)
      - [Online metrics](#online-metrics-2)
    - [Architectural Components](#architectural-components-2)
      - [Overview](#overview-4)
      - [Ad selection](#ad-selection)
      - [Ad prediction](#ad-prediction)
      - [Auction](#auction)
        - [Pacing](#pacing)
      - [Funnel model approach](#funnel-model-approach)
    - [](#)
  - [Entity Linking System](#entity-linking-system)
- [Recommender System](#recommender-system)

<br>

# Resources
- [Lessons on ML Platforms — from Netflix, DoorDash, Spotify, and more](https://towardsdatascience.com/lessons-on-ml-platforms-from-netflix-doordash-spotify-and-more-f455400115c7)
- https://www.youtube.com/c/ScottShiCS
- https://github.com/donnemartin/system-design-primer
- https://www.youtube.com/watch?v=ZgdS0EUmn70&list=PL73KFetZlkJSZ9vTDSJ1swZhe6CIYkqTL
- 斯坦福的 CS 329: https://stanford-cs329s.github.io/2021/syllabus.html
- [Rules of Machine Learning: Best Practices for ML Engineering -- Google](https://developers.google.com/machine-learning/guides/rules-of-ml#terminology)


## Rules of Machine Learning
- [Rules of Machine Learning: Best Practices for ML Engineering -- Google](https://developers.google.com/machine-learning/guides/rules-of-ml#terminology)

### Before ML

**Rule #1: Don’t be afraid to launch a product without machine learning**
- If you think that machine learning will give you a 100% boost, then a heuristic will get you 50% of the way there.

**Rule #2: First, design and implement metrics**

**Rule #3: Choose machine learning over a complex heuristic**
- A complex heuristic is unmaintainable
- A machine­-learned model is easier to update and maintain

<br>

### ML Phase I: Your First Pipeline

**Rule #4: Keep the first model simple and get the infrastructure right**
- Before anyone can use your fancy new machine learning system, you have to determine:
  - How to get examples to your learning algorithm.
  - A first cut as to what "good" and "bad" mean to your system.
  - How to integrate your model into your application
    - real-time?
    - pre-compute offline?

**Rule #5: Test the infrastructure independently from the machine learning**

**Rule #6: Be careful about dropped data when copying pipelines**

**Rule #7: Turn heuristics into features, or handle them externally**
- There are 4 ways you can use an existing heuristic:
  - Preprocess using the heuristic
  - Create a feature:
    - use raw value, bucketing, other transformation
    - combine with other features to create new features
  - Mine the raw inputs of the heuristic
  - Modify the label
- Do be mindful of the added complexity when using heuristics in an ML system. Using old heuristics in your new machine learning algorithm can help to create a smooth transition, but **think about whether there is a simpler way to accomplish the same effect**.

#### Monitoring

**Rule #8: Know the freshness requirements of your system**

**Rule #9: Detect problems before exporting models**

Rule #10: Watch for silent failures

Rule #11: Give feature columns owners and documentation

#### Your First Objective

Rule #12: Don’t overthink which objective you choose to directly optimize
- Some metrics are relevant, keep it simple and don’t think too hard about balancing different metrics when you can still easily increase all the metrics

Rule #13: Choose a simple, observable and attributable metric for your first objective
- The ML objective should be something that is easy to measure and is a **proxy** for the "true" objective
- Train on the simple ML objective, and consider having a "policy layer" on top that allows you to add additional logic (hopefully very simple logic) to do the final ranking.

Rule #14: Starting with an interpretable model makes debugging easier

Rule #15: Separate Spam Filtering and Quality Ranking in a Policy Layer

<br>

### ML Phase II: Feature Engineering

Rule #16: Plan to launch and iterate
- There are three basic reasons to launch new models:
  - You are coming up with new features.
  - You are tuning regularization and combining old features in new ways.
  - You are tuning the objective.

**Rule #17: Start with directly observed and reported features as opposed to learned features**

Rule #18: Explore with features of content that generalize across contexts
- ?

Rule #19: Use very specific features when you can.
- ?

**Rule #20: Combine and modify existing features to create new features in human­-understandable ways**
- The two most standard approaches are "**discretizations**" and "**crosses**".
  - `discretizations`: robust to outliers, more stable, faster
  - `crosses`: watch out the number of crosses features you create, large --> overfit

**Rule #21: The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have**
- Suppose you are working on a search ranking system:
  - If there are millions of different words in the documents and the query and you have `1000` labeled examples, then you should use a dot product between document and query features, `TF-IDF`, and a half-dozen other highly human-engineered features. 
    - **1000 examples, a dozen features.**
  - If you have a million examples, then intersect the document and query feature columns, using regularization and possibly feature selection. This will give you millions of features, but with regularization you will have fewer. 
    - Ten million examples, maybe a hundred thousand features.
  - If you have billions or hundreds of billions of examples, you can **cross the feature** columns with document and query tokens, using feature selection and regularization. 
    - You will have a billion examples, and 10 million features

**Rule #22: Clean up features you are no longer using**

#### Human Analysis of the System

**Rule #23: You are not a typical end user**
- avoid bias?

**Rule #24: Measure the delta between models**
- difference is small? --> no need to put into production
- difference is large? -->  make sure that the change is good

**Rule #25: When choosing models, utilitarian performance trumps predictive power**
- Your model may try to predict `CTR`. However, in the end, the key question is what you do with that prediction. If you are using it to rank documents, then **the quality of the final ranking** matters more than the prediction itself. 
- If you predict the probability that a document is `spam` and then have a cutoff on what is blocked, then the `precision` of what is allowed through matters more. 
- Most of the time, these two things should be in agreement: when they do not agree, it will likely be on a small gain. Thus, if there is some change that improves `log loss` but degrades `the performance of the system`, look for another feature. When this starts happening more often, it is time to revisit the objective of your model.

**Rule #26: Look for patterns in the measured errors, and create new features**
- Once you have examples that the model got wrong, look for trends that are outside your current feature set. 
- For instance, if the system seems to be demoting longer posts, then add post length. Don’t be too specific about the features you add. If you are going to add post length, don’t try to guess what long means, just **add a dozen features and the let model figure out what to do with them** (see Rule #21 )

**Rule #27: Try to quantify observed undesirable behavior**
- If your issues are measurable, then you can start using them as features, objectives, or metrics. The general rule is "measure first, optimize second".

**Rule #28: Be aware that identical `short-term` behavior does not imply identical `long-term` behavior**
- ?

#### Training-Serving Skew
Training-serving skew is a difference between performance during training and performance during serving. This skew can be caused by:
- A discrepancy between how you handle data in the training and serving pipelines.
- A change in the data between when you train and when you serve.
- A feedback loop between your model and your algorithm.

Rule #29: The best way to make sure that you train like you serve is to save the set of features used at serving time, and then **pipe those features to a log to use them at training time**.

**Rule #30: Importance-weight sampled data, don’t arbitrarily drop it!**
- ?

Rule #31: Beware that if you join data from a table at training and serving time, the data in the table may change.
- Say you join doc ids with a table containing features for those docs (such as number of comments or clicks). 
- Between training and serving time, features in the table may be changed. Your model's prediction for the same document may then differ between training and serving

**Rule #32: Re-use code between your training pipeline and your serving pipeline whenever possible**

**Rule #33: If you produce a model based on the data until January 5th, test the model on the data from January 6th and after**

Rule #34: In binary classification for filtering (such as spam detection or determining interesting emails), **make small short-term sacrifices in performance for very clean data**
- ??? 还能这样操作?

**Rule #35: Beware of the inherent skew in ranking problems**
- ??? 有点复杂

**Rule #36: Avoid feedback loops with positional features**
- position bias

**Rule #37: Measure Training/Serving Skew**


<br>

###

<br>

###

<br>


<br>

## ML Design Template
- https://www.mle-interviews.com/ml-design-template
- https://huyenchip.com/machine-learning-systems-design/research-vs-production.html#introduction-qzZkHeP

### Overeview
**Clarifying Requirements**
- How much data do we have? --> complexity of model
- Hardware Constraints? **Time Constraints**
- Inferece time vs Model accuracy trade-offs
- Model update frequency

**Metrics**
- Offline metrics: 
  - AUC, F1, MSE, R^2
- Online metrics:
  - CTR, CVR, Read time
- Non-functional metrics:
  - Training speed and scalability to very large datasets
  - **Extensibility to new techniques, new data sources**
  - Tools for easy training, debugging, evaluation and deployment

**Architecture**
- Data collection, label/target definition/identification
  - active learning, semi-supervised learning, weak-supervised learning
  - collect data for pre-training
  - hard positive & negative samples mining/generation
- Data preprocessing:
  - data cleaning, EDA, preliminary data sampling, feature selection
- Model building, de-buging, logging
- Model saving, prediction saving

### Data pipeline
- First, identify the target variable and how you would collect and label it (if necessary)
- Discuss features and possible feature crosses:
- **Data aggregation and cleaning**
- **Feature Engineering**
  - Train-test split strategy
    - 80/20 split
    - 5-fold CV
    - hold-out most recent one
  - Handle missing values or outliers
    - rule based
    - model based
  - **Balancing `positive` and `negative` training examples**
    - up-sampling, under-sampling
    - SMOTE
    - data augmentation
      - NLP: back-translation, paraphrase, entity/synonym replacement, add random noise to embeddings
  - **Apply different `scaling/encoding` techniques to different group of features**:
    - `numeric features`: standardization(robust: median), normalization(row-wise), min-max scaling, **bucketing**(stable, avoid outliers)
    - `categorical features`: one-hot encoding, label encoding(ordinal feature), target encoding(mean encoding, likelihood encoding, impact encoding), **cross-feature combination**, **embeddings**
  - **NLP feature engineering**:
    - tokenization: word, sub-word, character
    - lowercase, stemming, lemmatization
    - embeddings for word/sentence/doc, bag-of-words/tf-idf for sentence/doc
- Feature Selection(Optional): needed when there are not many data points or we want the model to be highly interpretable
- Additional Considerations: 
  - **Biases**: Are we sampling from a large enough subset of demographics, if not maybe we can group the largest and set others to be OOV demographics.
  - Any concerns with privacy/laws? We may need to `anonymize` or `remove` data depending on privacy concerns.

**MLOps for Data**: how to productionize above data steps, make it **reusable, scalable**
- **Storing data**:
  - object storage (images, videos, etc.) needs: Amazon S3, GCP Cloud Storage
  - Databases for `metadata` as well as `structured (tabular) data`: MySQL, Postgres, Oracle
  - **Feature Store** (store and access ML features) (Offline): FEAST, Amazon SageMaker Feature Store, Hopsworks Feature Store
  - Data Versioning/snapshot: DVC, Pachyderm
- **Data Ingesting and Transform**:
  - Ingesting:
    - Offline data --> can query your databases
    - Online data --> we need **high throughput and low latency** so we should use online streaming platforms like Apache Kafka and Apache Flume
  - Feature Transformation: Apache Spark, Tensorflow Transform
  - **Data drift detection/monitoring**:
    - feature/label distribution change
    - NLP vocab corpus change: too many OOV
- Orchestration Platforms:
  - Airflow
  - Kubernetes

### Model Training
- **Baseline model**: Typically one that does not require machine learning, just some rule based models (combinations)
  - For example, a good baseline like recommending the most popular products to the users. 
  - This "model" will always be easy to implement and all your other models should perform better than it
    - Random baseline
    - Human baseline
    - Simple heuristic baseline: business rules
- Traditional ML model: **give pros and cons**
  - Logistic Regression, Random Forest, XGB
- Deep neural network: data hungry models
  - Start simple and gradually add more components: RNN --> LSTM --> BERT
  - **Overfit a single batch for debugging**
  - Set a random seed for possible seeds: ensures consistency between different runs and reproduction
- Logging, hyper-parameters tuning, model selection
  - use config file/folder

**MLOPs for Modeling**:
- Repeatability of Experiments:
  - ML Flow
  - KubeFlow
- Parallelize hyper-parameter tuning: Google Cloud, Azure, AWS
- Model Versioning: DVC, Amazon SageMaker, Google Cloud AI Platform

### Model Serving
- Online A/B testing: talk about performing an A/B test using the online metric(s) mentioned earlier
- Where to run model inference: 
  - if we **run the model on the user's phone/computer** then it would use their `memory/battery` but `latency` should not be an issue
  - if we **store the model on the server**, we increase `latency` and `privacy` concerns but removes the burden of taking up `memory` and battery on the user's device.
- **Performance Monitoring**: some measurements that we should log would be 
  - error rates, time to return queries
  - metric scores
- **Biases and misuses of the model**: Does it propagate any gender and racial biases from the data?
- We should mention **how often we would retrain the model**. Some models need to be retrained every day, some every week and others monthly/yearly. Always discuss the pros and cons of the retraining regime you choose
  - online learning(no need training from scratch): 
    - RF(adding more trees)
    - Neural Network(I think any model that use SGD to optimize can be "fine-tuned" later on)

**ML OPs for Serving**:
- Store logs in a database like ElasticSearch, Logstash, GCP, AWS, Azure
- Logging analytic tools: Kibana, Splunk
- CI/CD: CircleCI, Travis CI
- **Deploying on Embedded and Mobile Devices**
  - Quantization: FP32 --> FP16?
  - **Reduced model size**:
    - layer params sharing
    - embedding fatorization: M x N --> M x K + K x N
    - select some layers in BERT to use
  - Knowledge Distillation
    - DistillBERT (for NLP)

<br>

# Grokking the Machine Learning Interview
## Overview
### Setting up a Machine Learning System
- **Setting up the problem**: ask questions to narrow down the problem space, the requirements of the system, and finally **arrive at a precise machine learning problem statement**
- **Understanding `scale` and `latency` requirements**
  - Latency requirements
  - Scale of the data
- **Defining metrics**
  - **Metrics for offline** testing
    - AUC, F1, MSE, R^2
  - Metrics for online testing: you may need both **component-wise** and **end-to-end** metrics, sometimes you may be asked to develop the ML system for a task that may be used as a component in other tasks
    - `component-wise` metric: NDCG for ranking
    - `end-to-end` metric: users’ engagement and retention rate, CTR, CVR, view time
  - **Non-functional metrics**:
    - Training speed and scalability to very large datasets
    - **Extensibility to new techniques, new data sources**
    - Tools for easy training, debugging, evaluation and deployment
- **Architecture discussion**: You need to think about the components of the system and how the data will flow through those components.
  - **Architecting for scale**: You can’t just build a complex ML model and run it for all ads in the system because it would take up a lot of time and resources. 
    - The solution is to use the **funnel approach**, where each stage will have fewer ads to process. This way, you can safely use complex models in later stages.
- **Offline model building and evaluation**:
  - `Training data generation`
    - In-house Human labeled data & Public data
    - Data collection through a user’s interaction with the pre-existing system
    - active learning, semi-supervised learning, weak-supervised learning
    - collect data for pre-training
    - hard positive & negative samples mining/generation
  - `Feature engineering`
    - Handle missing values or outliers
      - rule based
      - model based
    - **Balancing `positive` and `negative` training examples**
      - up-sampling, under-sampling
      - SMOTE
      - data augmentation
        - NLP: back-translation, paraphrase, entity/synonym replacement, add random noise to embeddings
    - **Apply different `scaling/encoding` techniques to different group of features**:
      - `numeric features`: standardization(robust: median), normalization(row-wise), min-max scaling, **bucketing**(stable, avoid outliers)
      - `categorical features`: one-hot encoding, label encoding(ordinal feature), target encoding(mean encoding, likelihood encoding, impact encoding), **cross-feature combination**, **embeddings**
  - `Model training`: select simpler models for the top of the funnel where data size is huge and more complex neural networks or trees based models for successive parts of the funnel.
    - use pre-trained model (transfer learning)
  - `Offline evaluation`: quickly test many different models so that we can **select the best one for online testing**, which is a slow process

<img src="imgs/RS_features.png" width = "1000" height = "250">

<br>

- **Online model execution and evaluation**: Depending on the type of problem, you may use both `component level` and `end-to-end` metrics
- **Iterative model improvement by model debugging**: Your model may perform well during offline testing, but the same increase in performance may not be observed during an online test. Here, you need to think about debugging the model to find out what exactly is causing this behavior.
  - Is the features’ distribution different during training and testing time? 
  - Moreover, after the first version of your model has been built and deployed, you still need to monitor its performance. 
    - You may observe a general failure from a decrease in AUC. 
    - You may note that the model is **failing in particular scenarios**. 
  - **hard positive & negative samples mining/generation**

<br>

## Practical ML Techniques/Concepts
### Performance and Capacity Considerations
Major performance and capacity discussions come in during the following two phases of building a machine learning system:
- `Training time`: How much training data and capacity is needed to build our predictor?
- `Evaluation time`: What are the [Service level agreement(SLA)](https://en.wikipedia.org/wiki/Service-level_agreement) that we have to meet while serving the model and capacity needs?

#### Complexities consideration for an ML system
Machine learning algorithms have three different types of complexities:
- `Training complexity`
- `Evaluation complexity`
- `Sample complexity`: the total number of training samples required to learn a target function successfully.

#### Comparison of `training` and `evaluation` complexities

#### `Performance` and `Capacity` considerations in large scale system
There can be many SLAs around availability and fault tolerance but for our discussion of designing ML systems, `performance` and `capacity` are the most important to think about when designing the system. 
  - `Performance` based SLA ensures that we return the results back within a given time frame (e.g. 500ms) for 99% of queries. 
  - `Capacity` refers to the load that our system can handle, e.g., the system can support 1000 `QPS` (queries per second).

To achieve the performance SLAs when we have huge data, e.g. 100 million documents to retrieve and rank, we need:
- `distributed systems`
- `funnel-based approach`: limited capacity, can't keep adding machines forever

#### Layered/funnel based modeling approach
In ML systems like **search ranking, recommendation, and ad prediction**, the `layered/funnel approach` to modeling is the right way to solve for scale and relevance while keeping performance high and capacity in check.

<img src="imgs/funnel_based_approach.png" width = "460" height = "320">

<br>

### Training Data Collection Strategies
#### Collection techniques
- **User’s interaction with pre-existing system (online)**
- **Human labelers (offline)**
  - Crowdsourcing: Amazon Mechanical Turk
  - Specialized labelers
  - Open-source datasets

#### Additional creative collection techniques
- **Build the product in a way that it collects data from user**
  - Pinterest: You want to show a personalized selection of `pins` to the new users to kickstart their experience. This requires data that would give you a semantic understanding of the `user` and the `pin`. This can be done by tweaking the system in the following way:
    - Ask `users` to name the board (collection) to which they save each `pin`. The name of the board will help to categorize the `pins` according to their content.
    - Ask `new users` to choose their interests in terms of the board names specified by `existing users`.
- **Creative manual expansion**
  - data augmentation:
    - CV: shift, rotate
    - NLP: back-translation, paraphrase, entity replacement
- **Data expansion using GANs**
  - CV: utilize GANs to convert images with sunny weather conditions to rainy weather conditions.

#### Train/Dev/Test splits
**Points to consider during splitting**
- While splitting training data, you need to ensure that you are **capturing all kinds of patterns in each split**.
- Most of the time, we are building models with the intent to **forecast the future**. Therefore, you need your splits to reflect this intent as well.

#### Quantity of training data

#### Training data filtering
It is essential to filter your training data since the model is going to learn directly from it. Any discrepancies in the data will affect the learning of the model.

- **Cleaning up data**
  - General data cleaning: handling missing data, outliers, duplicates and dropping out irrelevant features.
  - Apart from this, you need to analyze the data with regards to the given task to **identify patterns that are not useful**. 
    - For instance, consider that you are building a `search engine’s result ranking system`. 
    - Cases where the `user clicks` on a search result are considered `positive` examples. 
    - In comparison, those with `just an impression` are considered `negative` examples. 
    - You might see that the training data consist of a lot of **bot traffic** apart from the real user interactions. 
      - Bot traffic would **just contain `impressions` and no `clicks`**. This would introduce a lot of wrong negative examples!!!
- **Removing bias**: When we are generating training data through online user engagement, it may become biased. Removing this bias is critical. Let’s see why by taking the example of a movie recommender system like Netflix.
  - The pre-existing recommender is showing recommendations based on `popularity`. As such, the popular movies always appear first and new movies, although they are good, always appear later on as they have less user engagement. 
  - Due to the user’s time constraints, **he/she would only interact with the topmost recommendations** resulting in the generation of biased training data. The model hence trained, will continue considering the previous top recommendation to be the top recommendation this time too. Hence, the **“rich getting richer” cycle** will continue.
  - In order to break this cycle, we need to employ an **exploration technique** that explores the whole content pool (all movies available on Netflix). Therefore, we **show “randomized” recommendations** instead of “popular first” **for a small portion of traffic** for gathering training data. 
    - The users’ engagement with the randomized recommendations provides us with unbiased training data. 
    - This data really helps in removing the current **positional and engagement bias** of the system.
- **Bootstrapping new items**
  - Sometimes we are dealing with systems in which new items are added frequently. The new items may not garner a lot of attention, so we need to boost them to increase their visibility. 
  - For example, in the movie recommendation system, new movies face the cold start problem. We can **boost new movies by recommending them based on their similarity with the user’s already watched movies**, instead of waiting for the new movies to catch the attention of a user by themselves. 
  - Similarly, we may be building a system to display ads, and the new ads face the cold start problem. We can **boost them by increasing their relevance scores a little**, thereby artificially increasing their chance of being viewed by a person.

<br>

### Online Experimentation
For an ML system, “success” can be measured in numerous ways. 
- An advertising platform that uses a machine-learning algorithm to display relevant ads to the user. The success of this system can be measured using the **users’ engagement rate** with the advertisement and the **overall revenue** generated by the system
- Similarly, a search ranking system might take into account correctly ranked search results on **SERP** as a metric to claim to be a successful search engine. 

#### Running an online experiment(A/B test)
We can formulate the following two hypotheses for the A/B test:
- **The null hypothesis**: $H_0$ is when the design change will not have an effect on variation. 
  - If we fail to reject the null hypothesis, we should not launch the new feature.
- **The alternative hypothesis**: $H_1$ is alternate to the null hypothesis whereby the design change will have an effect on the variation. 
  - If the null hypothesis $H_0$ is rejected, then we accept the alternative hypothesis and we should launch the new feature. Simply put, the variation will go in production.

Before statistically analyzing the results, a [power analysis test](https://en.wikipedia.org/wiki/Power_of_a_test) is conducted to determine how much overall traffic should be given to the system, i.e., the `minimum sample size` required to see the impact of conversion. Half of the traffic is sent to the control, and the other half is diverted towards the variation.

#### Measuring results: Computing statistical significance
- [P-value](https://en.wikipedia.org/wiki/P-value) is used to help determine the statistical significance of the results. In interpreting the p-value of a significance test, a significance level ($\alpha$) must be specified.
> The significance level is a boundary for specifying a statistically significant finding when interpreting the p-value. A commonly used value for the significance level is 5% written as 0.05.
- The result of a significance test is claimed to be “statistically significant” if the p-value is less than the significance level.
  - $p <= \alpha$: reject $H_0$ - launch the new feature
  - $p > \alpha$: fail to reject $H_0$ - do not launch the new feature
- If an A/B test is run with the outcome of a significance level of 95% (p-value <= 0.05), there is a **5% probability that the variation that we see is by chance**

#### Measuring long term effects
In some cases, we need to be more confident about the result of an A/B experiment **when it is overly optimistic**.

**Back Testing**
- Let’s assume that variation improved the overall system performance by 5% when the expected gain was 2%.
- In the case of the ads prediction system, we can say that the rate of user engagement with the ad increased by 5% in variation (system B). This surprising change puts forth a question. **Is the result overly optimistic?** To confirm the hypothesis and be more confident about the results, we can perform a [backtest](https://en.wikipedia.org/wiki/Backtesting). Now we change criteria, system A is the previous system B, and vice versa.
- We will check all potential scenarios while backtesting:
  - Do we lose gains? 
  - Is the gain caused by an A/B experiment equal to the loss by B/A experiment? 
    - Assume that the A/B experiment gave a gain of 5% and B/A experiment gave a loss of 5%. 
    - This will ensure that the changes made in the system improved performance.

  <img src="imgs/Backtest.png" width = "400" height = "240">

<br>

**Long-running A/B tests**
- In a few experiments, one key concern could be that the experiment can have a **negative long term impact** since we do A/B testing for only a short period of time. Will any negative effects start to appear if we do a long term assessment of the system subject to variation?
- For example, suppose that for the ad prediction system, the revenue went up by 5% when we started showing more ads to users but this had no effect on user retention. Will users start leaving the platform if we show them significantly more ads over a longer period of time? To answer this question, we might want to have a long-running A/B experiment to understand the impact.
- The long-running experiment, which measures long-term behaviors, can also be done via a backtest. We can launch the experiment based on initial positive results while continuing to run a long-running backtest to measure any potential long term effects. If we can notice any significant negative behavior, we can revert the changes from the launched experiment.

<br>

### Embeddings

#### Text embeddings
**Word2vec**
- `CBOW(Continuous Bag Of Words)`: predict the center word from its surrounding words
- `Skipgram`: predict surrounding words from the center word.
- **Example**
  - Let’s assume that we want to predict whether a `user` is interested in a particular `document` given the documents that they have previously read. 
  - One simple way of doing this is to represent the `user` by taking the `mean of the Word2vec embeddings of document titles` that they haved engaged with. 
  - Similarly, we can represent the `document` by the `mean of its title term embeddings`. 
  - We can simply take the dot product of these two vectors and use that in our ML model.
  - Another way to accomplish this task is to simply pass the user and the document embedding vector to a neural network to help with the learning task.

**Context-based embeddings**
- Once trained, Word2vec embeddings have a fixed vector for every term. So, a Word2vec embedding doesn’t consider the context in which the word appears to generate its embedding. However, words in a different context can have very different meanings.
- Two popular architectures used to generate word context-based embedding are:
  - Embeddings from Language Models (**ELMo**): builds bi-directional language model using LSTM
    - feature-based fashion
  - Bidirectional Encoder Representations from Transformers (**BERT**): uses Transformer encoder
    - fine-tuning fashion

#### Visual embedding
**Auto-encoders**
- Auto-encoders use neural networks consisting of both an `encoder` and a `decoder`. 
  - `encoder`: first learn to compress the raw image pixel data to a small dimension
  - `decoder`: then try to de-compress it and re-generate the same input image. 
  - The last layer of `encoder` determines the dimension of the embedding, which should be sufficiently large to capture enough information about the image so that the `decoder` can decode it.
- The combined encoder and decoder tries to minimize the difference between original and generated pixels, using backpropagation to train the network
- Once we have trained the model, we only use the encoder (first N network layers) to generate embeddings for images.

**Visual supervised learning tasks**
- Visual supervised learning tasks such as `image classification` or `object detection`, are generally set up as `convolution layers, pooling layers`, and `fully connected network layers`, followed by `final classification(softmax) layers`. 
  - The penultimate layer before `softmax` captures all image information in a vector such that it can be used to classify the image correctly. So, we can use the penultimate layer value of a pre-trained model as our image embedding.
- An example of image embedding usage could be to `find images similar to a given image`.
- Another example is an `image search problem` where we want to find the best images for given text terms, e.g. query “cat images”. In this case, image embedding along with query term embedding can help refine search relevance models.

#### Learning embeddings for a particular learning task
- Most of our discussion so far has been about training a general entity embedding that can be used for any learning task. However, we can also embed an entity as part of our learning task. The advantage of this embedding is a specialized one for the given prediction task

#### Network/Relationship-based embedding
- Most of the systems have multiple entities, and these entities interact with each other. For example:
  - Pinterest has `users` that interact with `Pins`
  - YouTube has `users` that interact with `videos`
  - Twitter has `users` that interact with `tweets`
  - Google search has both `queries` and `users` that interact with `web results`.
- We can think of these interactions as relationships in a **graph** or resulting in **interaction pairs**. For the above example, these pairs would look like:
  - `(User, Pin)` for Pinterest
  - `(User, Video)` for YouTube
  - `(User, Tweet)` for Twitter
  - `(Query, Webpage)` for Search
  - `(Searcher, Webpage)` for Search
- In all the above scenarios, the retrieval and ranking of results for a particular `user` (or `query`) are mostly about predicting how close they are. Therefore, having an embedding model that projects these documents in the same embedding space can vastly help in the retrieval and ranking tasks of recommendation, search, feed-based, and many other ML systems.
- We can generate embeddings for both the above-discussed pairs of entities in the same space by creating a **two-tower neural network model** that tries to encode each item using their raw features. The model optimizes the inner product loss such that `positive` pairs from entity interactions have a higher score and `random(negative)` pairs have a lower score. 
  - a lot of options for loss functions
$$
L o s s=\max \left(\sum_{(u, v) \in A} \operatorname{dot}(u, v)-\sum_{(u, v) \notin A} \operatorname{dot}(u, v)\right)
$$

<img src="imgs/two_tower.png" width = "500" height = "300">

<br>

### Transfer Learning
#### Overview

**Motivation**

<br>

### Model Debugging and Testing
There are two main phases in terms of the development of a model that we will go over:
- Building the first version of the model and the ML system.
- Iterative improvements on top of the first version as well as debugging issues in large scale ML systems.

#### Building model v1
Important Steps of build the 1st version of the model:
- We begin by identifying a business problem in the first phase and mapping it to a machine learning problem.
- We then go onto explore the training data and machine learning techniques that will work best on this problem.
- Then we train the model given the available data and features, play around with hyper-parameters.
- Once the model has been set up and we have early offline metrics like accuracy, precision/recall, AUC, etc., we continue to play around with the various features and training data strategies to improve our offline metrics.
- If there is already a heuristics or rule-based system in place, our objective from the offline model would be to perform at least as good as the current system, e.g., for ads prediction problem, we would want our ML model AUC to be better than the current rule-based ads prediction based on only historical engagement rate.

<img src="imgs/ML_model_workflow.png" width = "800" height = "220">

<br>

#### Deploying and debugging v1 model
In our first attempt to take the model online, i.e., enable live traffic, might not work as expected and results don’t look as good as we anticipated offline. Let’s look at a few failure scenarios that can happen at this stage and how to debug them.

**Change in feature distribution**
- The change in the feature distribution of training and evaluation set, take an Entity linking system as an example
  - training: Wikipedia dataset
  - online inference: Wikipedia articles + research papers
- A significant change in incoming traffic because of seasonality, consider an example of a search system trained using data for the last 2 weeks of December, i.e., mostly holiday traffic. 
  - If we deploy this system in January, the queries that it will see will be vastly different than what it was trained on and hence not performing as well as we observed in our offline validation.

**Feature logging issues**
- Let’s assume that we have one important feature/signal for our model that’s based on **historical advertiser ad impressions**. 
  - During `training`, we compute this feature by using the `last 7 days’ impression`. 
  - But, the logic to compute this feature at model `evaluation` time uses the `last 30 days of data` to compute advertiser impressions. 
  - Because of this feature computed differently at training time and evaluation time, it will result in the model not performing well during online serving!

**Overfitting and Under-fitting**

#### Iterative model improvement
The best way to iterative improve model quality is to start looking at failure cases of our model prediction and using that come up with the ideas that will help in improving model performance in those cases.

**Missing important feature**
- On debugging, we figure out that the user has previously watched two movies by the same actor, so adding a feature on previous ratings by the user for this movie actor can help our model perform better in this case.

**Insufficient training examples**
- extreme data imbalance over certain categories

#### Debugging large scale systems
The following are a few key steps to think about iterative model improvement for large scale end to end ML systems:
- **Identify the component**
  - This accounts for finding the architectural component resulting in a high number of failures in our failure set. Suppose that in the case of the **search ranking system**, we have opted for a **layered model approach**.
  - In order to see the cause of failure, we will look at each layers’ performance to understand the opportunity to significantly improve the quality of our search system. Let’s assume that our search system has **two key components**
    - (1) **Document selection**: focus on ensuring that all the top relevant documents get selected for the query
    - (2) **Ranking of selected documents**: ensures that our rank order is correct based on the relevance of the top 100 documents.
  - If 80% of our overall search system failures are because of the ideal document not being selected in candidate selection component, we will debug the model deeply in that layer to see how to improve the quality of that model. Similarly, if ideal documents are mostly selected but are ranked lower by our ranking component, we will invest in finding the reason for failures in that layer and improve the quality of our ranking model.

- **Improve the quality of component**
  - Some of the model improvement methods that we have discussed above like **adding more training data, features, modeling approach in case of overfitting and underfitting will still be the same** once we identify the component that needs work, e.g., if we identify that the candidate selection layer needs improvement in our search, we will try to see missing features, add more training data or play around with ML model parameters or try a new model.

<br>

## Search Ranking
### Problem Statement
- Design a search relevance system for a search engine

#### Clarifying questions
- Let’s clarify the problem statement by specifying three aspects: `scope`, `scale`, and `personalization`.

**Problem scope**
- The interviewer’s question is really broad. Your best bet is to avoid ambiguities and ask the interviewer as many questions as you can. 
- So, your first question for the interviewer would be something like the following:
> Is it a **general search engine** like Google or Bing or a **specialized search engine** like Amazon products search?
- This scoping down of the problem is critical as you dive into finding the solutions. For the sake of this chapter, we will assume that you are working towards finding relevant results using **a general search engine** like Google search or Bing search, 
- Finally, the problem statement can be precisely described as:
> Build a generic search engine that returns relevant results for queries like “Richard Nixon”, “Programming languages” etc.
- This will require you to build a machine learning system that provides the most relevant results for a search query by ranking them in order of relevance. 

**Scale**
- Once you know that you are building a general search engine, it’s also critical to determine the scale of the system. A couple of important questions are:
  - How many websites exist that you want to enable through this search engine?
  - How many requests per second do you anticipate to handle?
- We will assume that you have **billions of documents** to search from, and the search engine is getting around **10K queries per second (QPS)**.

**Personalization**
- Another important question that you want to address is whether the searcher is a **logged-in user or not**. This will define the level of personalization that you can incorporate to improve the relevance of our results. 
- You will assume that **the user is logged in** and you have access to their profile as well as their historical search data.

<br>

### Metrics
#### Online metrics
- `Click-Through Rate(CTR)`: the ratio of clicks to impressions
- `Successful Session Rate`: 
  - CTR might include hort clicks where the searcher only looked at the resultant document and clicked back immediately. You could solve this issue by filtering your data to **only successful clicks**, i.e., to only consider clicks that have a long dwell time.
> Dwell time is the length of time a searcher spends viewing a webpage after they’ve clicked a link on a **search engine result page (SERP)**.

**Caveat**
- Another aspect to consider is `zero-click` searches.
> Zero-click searches: A SERP may answer the searcher’s query right at the top such that the searcher doesn’t need any further clicks to complete the search.
- For example, a searcher queries “einstein’s age”, and the SERP shows an **excerpt(summary)** from a website in response, as shown below:
- The searcher has found what they were looking for **without a single click**! The click-through rate would not work in this case (but your definition of a **successful session** should definitely include it). We can fix this using a simple technique shown below.

<img src="imgs/Zero-click-search.png" width = "700" height = "400">

<br>

- **Time to success**
  - Until now, we have been considering a single query-based search session. However, it may span over several queries. For example, the searcher initially queries: “italian food”. They find that the results are not what they are looking for and make a more specific query: “italian restaurants”. Also, at times, the searcher might have to go over multiple results to find the one that they are looking for.
  - Ideally, **you want the searcher to go to the result that answers their question in the minimal number of queries and as high on the results page as possible**. So, `time to success` is an important metric to track and measure search engine success.
  - The shorter the better?
> Note: For scenarios like this, a **low number of queries per session** means that your system was good at guessing what the searcher actually wanted despite their poorly worded query. So, in this case, we should consider a low number of queries per session in your definition of a successful search session.

#### Offline metrics
Trained human raters need to rate the relevance of the query results objectively, these ratings are then aggregated across a query sample to serve as the ground truth

##### NDCG

Let’s see `Normalized Discounted Cumulative Gain (NDCG)` in detail as it’s a critical evaluation metric for any ranking problem.

- https://zhuanlan.zhihu.com/p/136199536
- 在 Ranking 的应用及其相关的评测中作如下假设：
  - 高相关性的文档出现的位置越靠前，指标会越高。
  - 高相关性的文档比一般相关性的文档更影响最终的指标得分。
- **CG(Cumulative Gain)**: 只考虑到了相关性的关联程度，**没有考虑到位置的因素**，它是一个搜素结果相关性分数的总和:
$$
\mathrm{CG}_{\mathrm{p}}=\sum_{i=1}^{p} r e l_{i}
$$
  - 假设搜索“乒乓球”结果，最理想的结果是：`B1, B2, B3`。而出现的结果是 `B3, B1, B2` 的话，CG的值是没有变化的，因此需要下面的DCG
- **DCG(Discounted cumulative gain)**: penalize the search engine’s ranking if highly relevant documents (as per ground truth) appear lower in the result list, 两个思想
  - 高关联的结果比一般关联的结果更影响最终的指标得分
  - 高关联的结果出现在更靠前的位置的时候，指标会更高
  - **DCG计算**: 就是在每一个CG的结果上除以一个折损值，为什么要这么做呢？目的就是为了让排名越靠前的结果越能影响最后的结果。假设排序越往后，价值越低。到第i个位置的时候，它的价值是 `1/log2(i+1)`，那么第i个结果产生的效益就是 `reli * 1/log2(i+1)`:
    - 第二个公式应用更广泛
$$
\mathrm{DCG}_{\mathrm{p}}=\sum_{i=1}^{p} \frac{r e l_{i}}{\log _{2}(i+1)}, or \quad \mathrm{DCG}_{\mathrm{p}}=\sum_{i=1}^{p} \frac{2^{rel_{i}} - 1}{\log _{2}(i+1)}
$$
- **NDCG(Normalized Discounted Cumulative Gain)**: 归一化折损累计增益
  - **The length of the result list varies from query to query**
  - 由于搜索结果随着检索词的不同，返回的数量是不一致的，而DCG是一个累加的值，没法针对两个不同的搜索结果进行比较，因此需要归一化处理，这里是除以 `IDCG(Ideal Discounted Cumulative Gain)` or max DCG:
  $$\begin{aligned}
  \mathrm{IDCG}_{\mathrm{p}} &= \sum_{i=1}^{|R E L|} \frac{2^{\text {rel }_{i}}-1}{\log _{2}(i+1)} \\
  \mathrm{NDCG}_{\mathrm{p}} &= \frac{D C G_{p}}{I D C G_{p}}
  \end{aligned}
$$
  - 其中 $|REL|$ 表示，结果按照相关性从大到小的顺序排序，取前$p$个结果组成的集合。也就是**按照最优的方式**对结果进行排序
- **例子**：
  - 假设搜索回来的5个结果，其相关性分数分别是 `3、2、3、0、1、2`
  - `CG = 3+2+3+0+1+2`，可以看到只是对相关的分数进行了一个关联的打分，并没有召回的所在位置对排序结果评分对影响。
  - `DCG = 3+1.26+1.5+0+0.38+0.71 = 6.86`
  - 接下来我们归一化，归一化需要先结算 IDCG，假如我们实际召回了8个物品，除了上面的6个，还有两个结果，假设第7个相关性为3，第8个相关性为0。那么在理想情况下的相关性分数排序应该是：`3、3、3、2、2、1、0、0`
    - `IDCG@6 = 3+1.89+1.5+0.86+0.77+0.35 = 8.37`
    - `NDCG@6 = 6.86/8.37 = 81.96%`
- **Caveat**
  - `NDCG` **does not penalize irrelevant search results**, which had zero relevance according to the human rater.
  - As a remedy, the human rater could assign a **negative relevance score** to that document.

<br>

### Architectural Components

<img src="imgs/search_ranking_arch.png" width = "660" height = "350">

<br>

#### Query rewriting
We use query rewriting to increase `recall`, i.e., to retrieve a larger set of relevant results. Query rewriting has multiple components which are mentioned below.
- **Spell checker**
  - Spell checking queries is an integral part of the search experience and is assumed to be a necessary feature of modern search. 
  - Spell checking allows you to fix basic spelling mistakes like “itlian restaurat” to “italian restaurant”.
- **Query expansion**
  - Query expansion improves search result retrieval by adding terms to the user’s query. 
  - Essentially, these additional terms minimize the mismatch between the searcher’s query and available documents, for the query “italian restaurant”, we should expand “restaurant” to food or recipe to look at all potential candidates (i.e., web pages) for this query.
> The reverse, i.e., query relaxation, serves the same purpose. For example, a search for “good italian restaurant” can be relaxed to “italian restaurant”.

#### Query understanding
This stage includes figuring out **the main intent** behind the query, e.g., the query “gas stations” most likely has a **local intent** (an interest in nearby places) and the query “earthquake” may have a **newsy intent**. Later on, this intent will help in selecting and ranking the best documents for the query. 

#### *Document selection
- Document selection is more focused on `recall`. It uses a simpler technique to sift through `billions` of documents on the web and retrieve documents that have the potential of being relevant.
- Ranking these selected documents in the right order isn’t important at this point. We let the `ranking component` worry about finding out “exactly” how relevant (`precision`) each selected document is and in what order they should be displayed on the SERP.

#### *Ranker
- The ranker will actively utilize machine learning to find the best order of documents (this is also called **Learning to Rank**).
- If the number of documents from the document selection stage is significantly large (more than `10k`) and the amount of incoming traffic is also huge (more than `10k` QPS or queries per second), you would want to have **multiple stages of ranking with varying degrees of complexity** and model sizes for the ML models.
  - **Multiple stages in ranking** can allow you to only utilize complex models at the very last stage where ranking order is most important
  - For example, one configuration can be that your document selection returns `100k` documents, and you pass them through **two stages of ranking**. 
    - In stage one, you can use fast (nanoseconds) linear ML models to rank them.
    - In stage two, you can utilise computationally expensive models (like deep learning models) to find the most optimized order of top 500 documents given by stage one.

#### Blender
- Blender gives relevant results from various search verticals, like, `images`, `videos`, `news`, `local results`, and `blog posts`.
- **Filter** will further screen `irrelevant` or `offensive` results to get good user engagement

#### Training data generation
- It takes online user engagement data from the SERP displayed in response to queries and generates `positive` and `negative` training examples. The training data generated is then fed to the machine learning models trained to rank search engine results.

#### Layerd model approach
- Document selection --> Ranker 1 --> Ranker 2 --> Blender --> Filter

<br>

### Document Selection
#### Overview
- From the one-hundred `billion` documents on the internet, we want to retrieve the top one-hundred `thousand` that are relevant to the searcher’s query by using **information retrieval techniques**.
- Let’s get some terminologies out of the way before we start:
  - **Documents**
    - Web-pages
    - Emails
    - Books
    - News stories
    - Scholarly papers
    - Text messages
    - Word™ documents, Powerpoint™ presentations
    - PDFs
    - Patents, etc.
  - **Inverted index**: an index data structure that stores a mapping from `content`, such as `words` or `numbers`, to its `location` in a set of documents.

<img src="imgs/Inverted_index.png" width = "600" height = "350">

<br>

#### Selection criteria
Our document selection criteria would then be as follows:

<img src="imgs/Selection_criteria.png" width = "550" height = "200">

<br>

We will go into the `index` and retrieve all the documents based on the above selection criteria. While we would check whether each of the documents matches the selection criteria, we would also be assigning them a **relevance score** alongside

#### Relevance scoring scheme
Let’s see how the relevance score is calculated. One basic scoring scheme is to utilize a simple **weighted linear combination** of the factors involved. The weight of each factor depends on its importance in determining the relevance score. Some of these factors are:
- **Terms match**: The term match score contributes with `0.5` weight to the document’s relevance score.
  - Our query contains multiple terms. We will use the **inverse document frequency (IDF)** score of each term to weigh the match. The match for important terms in the query weighs higher. 
  - For instance, the term match for `“italian”` may have more weight in the total contribution of term match to the document’s relevance score.
- **Document popularity**
- **Query intent match**: The query intent component describes the intent of the query. 
  - For above query, the component may reveal that there is a very strong `local intent`
- **Personalization match**: This reflects how well a document meets the searcher’s individual requirements based on a lot of aspects, such as the searcher’s `age`, `gender`, `interests`, and `location`.

<img src="imgs/Relevance_scoring_scheme.png" width = "660" height = "330">

<br>

<br>

### Feature Engineering
4 important factors for search are:
- **Searcher**
- **`Query`**
- **`Document`**
- **Context**: search history, searcher’s age, gender, location, previous queries and the time of day.

Let’s go over the characteristics of these actors and their interactions to generate meaningful features/signals for your machine learning model. A subset of these features is shown below.

<img src="imgs/search_features.png" width = "960" height = "280">

<br>

#### Searcher-specific features
- Assuming that the searcher is logged in, you can tailor the results according to their age, gender and interests by using this information as features for your model.

#### Query-specific features
- **Query historical engagement**
  - For **relatively popular queries**, historical engagement can be very important. You can use **query’s prior engagement** as a feature. 
  - For example, let’s say the searcher queries `“earthquake”`. We know from historical data that this query results in engagement with `“news component”`, i.e. most people who searched “earthquake”, were looking for news regarding a recent earthquake. Therefore, you should consider this factor while ranking the documents for the query.
- **Query intent**
  - The “query intent” feature enables the model to identify **the kind of information** a searcher is looking for when they type their query. The model uses this feature to assign a higher rank to the documents that match the query’s intent. 
  - For instance, if a person queries “pizza places”, the **intent here is local**. Therefore, the model will give high rank to the pizza places that are located near the searcher.
  - A few examples of query intent are `news, local, commerce`, etc.
  - We can get query intent from the **query understanding component** -- classification model?

#### Document-specific features
- **Page rank**
  - The rank of a document can serve as a feature. To estimate the relevance of the document under consideration, we can look at **the number and quality of the documents that link to it**.
- **Document engagement radius**
  - The document engagement radius can be another important feature. A document on a coffee shop in Seattle would be more relevant to people living within a **ten-mile radius** of the shop. However, a document on the Eiffel Tower might interest people all around the world. Hence, in case our query has a local intent, we will choose the document with **the local scope of appeal** rather than that with a global scope of appeal.

#### Context-specific features
We can extract features from the context of the search.
- **Time of search**
  - A searcher has queried for restaurants. In this case, a contextual feature can be the time of the day. This will allow the model to display restaurants that are open at that hour.
- **Recent events**
  - The searcher may appreciate any recent events related to the query. For example, upon querying “Vancouver”, the results included:
- **Previous Queries**:
  - To provide relevant results for a query, looking at the nature of the previous queries can also help

<img src="imgs/prev_query_context.png" width = "400" height = "270">

<br>

#### Searcher-document features
We can also extract features by considering both the `searcher` and the `document`.
- **Distance**
  - For queries inquiring about nearby locations, we can use the `distance between the searcher` and the `matching locations` as a feature to measure the relevance of the documents.
- **Historical engagement**
  - Another interesting feature could be the searcher’s historical engagement with the `result type of the document`. 
  - For instance, if a person has engaged with `video documents` more in the past, it indicates that video documents are generally more relevant for that person.
  - Historical engagement with `a particular website or document` can also be an important signal as the user might be trying to “re-find” the document.

#### Query-document features#
Given the query and the document, we can generate tons of signals.
- **Text Match**
  - Text match can not only be in the `title` of the document, but it can also be in the `metadata` or `content` of a document.
- **Bigram & Trigram + TF-IDF**
  - We can also look at data for each unigram and bigram for text match between the query and document. 
  - For instance, the query: `“Seattle tourism guide”` will result in three unigrams:
    - `Seattle`
    - `tourism`
    - `guide`
  - These unigrams may match different parts of the document. Similarly, we can check the match for the `bigram` and the full `trigram`, as well. All of these text matches can result in **multiple text-based features** used by the model.
  - **TF-IDF** match score can also be an important relevance signal for the model. It is a similarity score based on a text match between the query terms and the document. 
    - **TF(term frequency)** incorporates the importance of each term for the document
    - **IDF(inverse document frequency)** tells us about how much information a particular term provides.
- **Query-document historical engagement**
  - `Click rate`
    - We want to see users’ historical engagement with documents shown in response to a particular query. The click rates for the documents can help in the ranking process. 
    - For example, we might observe across people’s queries on “Paris tourism” that the click rate for the “Eiffel tower website” is the highest. So, the model will develop the understanding that whenever someone queries “Paris tourism”, the document/website on Eiffel tower is the most engaged with. It can then use this information in the ranking of documents.
- **Embeddings**
  - We can use embedding models to represent the `query` and `documents` in the form of vectors.
  - The embedding model generates vectors in such a manner that if a `document` is on the same `topic/concept` as the `query`, its vector is similar to the query’s vector. 
  - We can use this characteristic to create a feature called `“embedding similarity score”`. The higher the similarity score, the more relevant a document is for a query.

<br>

### Training Data Generation
#### Data generation for pointwise approach
- **Pointwise approach**: 
  - In this approach of model training, the training data consists of relevance scores for each document. The loss function looks at the score of one document at a time as an absolute ranking. Hence the model is trained to predict the relevance of each document for a query, **individually**. 
  - The final ranking is achieved by simply sorting the result list by these document scores.
- While adopting the pointwise approach, our ranking model can make use of classification algorithms.
- For instance, if we aim to simply classify a document as `relevant` or `irrelevant`, the relevance score will be `0` or `1`. This will allow us to approximate the ranking problem by a binary classification problem.

##### Positive and negative training examples
- We are essentially predicting user engagement towards a document in response to a query. A relevant document is one that successfully engages the searcher.
- For instance, we have the searcher’s query: “Paris tourism”, and the following results are displayed on the SERP in response:
  - Paris.com
  - Eiffeltower.com
  - Lourvemusuem.com
- We are going to label our data as `positive/negative` or `relevant/irrelevant`, keeping in mind the metric **successful session rate**
  - CTR is also ok
- **Assumption**
  - Let’s assume that the searcher did not engage with Paris.com but engaged with Eiffeltower.com. 
  - Upon clicking on Eiffeltower.com, they spent `two minutes` on the website and then signed up. After signing up, they went back to the SERP and clicked on Lourvemusuem.com and spent `twenty seconds` there.
  - This sequence of events generates three rows of training data. 
    - “Paris.com” would be a `negative` instance because it only had a view with no engagement. The user skipped it and engaged with the other two links, which would become `positive` examples.
- **Caveat: Less `negative` examples**
  - A question may arise that if the user engages with only the first document on the SERP, we may never get enough negative examples to train our model. Such a scenario is pretty common. 
  - To remedy it, we use **random negative examples**. For example, all the documents displayed on the $50^{th}$ page of Google search results can be considered negative examples.
  - User engagement patterns may **differ throughout the week**. 
    - For instance, the engagement on weekdays may differ from the weekend. Therefore, we will **use a week’s queries to capture all patterns** during training data generation

#### Data generation for pairwise approach
- **Pairwise approach**: 
  - This approach differs from the pointwise approach during model training. Here, the loss function looks at **the scores of document pairs** as an inequality instead of the score of a single document. This enables the model to learn to **rank documents according to their relative order** which is closer to the nature of ranking. 
  - Hence, the model tries to predict the document scores such that **the number of `inversions` in the final ranked results is minimum**. 
    - `Inversions` are cases where **the pair of results are in the wrong order** relative to the ground truth.
  - We are going to explore two methods of training data generation for the pairwise approach to the ranking problem:
    - The use of human raters
    - The use of online user engagement

##### Human raters (offline method)
- Let’s assume that the human rater will be presented with `100,000` queries, each having `10` results. They will then be asked to rate the results for each query. The rater might rate a document as:

| Rating    | Ranking |
| :-------- | :------ |
| perfect   | 4       |
| excellent | 3       |
| good      | 2       |
| fair      | 1       |
| bad       | -1      |

##### User-engagement (online method)
- The user’s interaction with the results on the SERP will translate to ranks based on the type of interaction.
- One intuitive idea can be to use online engagement to generate pairwise session data, e.g., one session can have **three types of engagements** varying from higher degrees of relevant actions to lower.
- For instance, for the query: “Paris tourism”, we get the following three results on the SERP:
  - Paris.com: only has an impression with no click, which would translate to `label 0`
  - Eiffeltower.com: The searcher signs up on Eiffeltower.com. It will be rated as `perfect` and assigned the `label 4`
  - Lourvemusuem.com: will be rated as `excellent` and assigned the `label 3`, as the searcher spends twenty seconds exploring the website.

<br>

### Ranking
#### Overview
- You want to build a model that will display the most relevant results for a searcher’s query. **Extra emphasis should be applied to the relevance of the top few results** since a user generally focuses on them more on the SERP.
- **Learning to Rank (LTR)**: A class of techniques that applies supervised machine learning (ML) to solve ranking problems. The `pointwise` and `pairwise` techniques that we will apply fall under this class.
- **Stage-wise approach**
  - `Document selection`: 100 million documents --> 100K relevant results/pages
  - `Ranking Stage 1`: 100K relevant results/pages --> 500 relevant documents
    - ensure that the topmost relevant results are forwarded to `Stage 2`
    - focus: `recall` of top documents
  - `Ranking Stage 2`: rank 500 documents
    - ensure the topmost relevant documents are in correct order
    - focus: `precision` of top documents

#### Stage 1 (Logistic regression)
- As we try to limit documents in this stage from a large set to a relatively smaller set, it’s important that we don’t miss out on highly relevant documents for the query from the smaller set. 
- This can be achieved with the **pointwise approach**. The problem can be approximated with the **binary classification** of results as `relevant` or `irrelevant`.

**Logistic regression**
- A relatively less complex linear algorithm, like `logistic regression` or small `MART`(Multiple additive regression trees) model, is well suited for scoring a large set of documents. The ability to score each document extremely quickly (microseconds or less) for the fairly large document pool at this stage is super critical.
- Use metric like AUC to determine the optimal model

#### Stage 2 (LambdaMART, LambdaRank)
- The main objective of the stage 2 model is to **find the optimized rank order**.
- This is achieved by changing the objective function from a single **pointwise objective** (`click`, `session success`) to a **pairwise objective**. 
- Pairwise optimization for learning to rank means that the model is not trying to minimize the classification error but rather trying to **get as many pairs of documents in the right order as possible**.

**LambdaMART**
- `LambdaMART` is a variation of `MART` where we **change the objective to improve pairwise ranking**, as explained above. 
- **Tree-based algorithms** are generally able to generalize effectively using a moderate set of training data. Therefore, if your training data is limited to a few million examples, this definitely will be the best choice to use in pairwise ranking in the second stage.
- If we are optimizing for **offline `NDCG`** (based on **human-rated data**), then this is definitely one of the best options.

**LambdaRank**
- LambdaRank is a **neural network-based approach utilizing pairwise loss** to rank the documents. 
- Neural network-based models are relatively slower (given the large number of computations based on width and depth) and need more training data. So training data size and capacity are key questions before selecting this modeling approach. 
- The **online training data generation method** for the pairwise approach can generate ranking examples for a popular search engine in abundance. 
- Your training data contains pairs of documents $(i, j)$, where $i$ ranks higher than $j$. Let's look at the Lambda rank model's learning process. 
  - Suppose we have to rank two documents $i$ and $j$ for a given query. We feed their corresponding feature vectors $x_{i}$ and $x_{j}$ to the model, and it gives us their relevance scores, i.e., $s_{i}$ and $s_{j}$. 
  - The model should compute these scores ($s_{i}$ and $s_{j}$) such that the probability of document $i$ being ranked higher than document $j$ is close to that of the ground truth. 
  - The optimization function tries to **minimize the inversions in the ranking**.
- Both `LambdaMART` and `LambdaRank` are very well explained in this paper [From RankNet to LambdaRank to
LambdaMART: An Overview](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf).
- We can calculate the `NDCG` score of the ranked results to compare the performance of different models.

<br>

### Filtering Results
#### Result set after ranking
The result set might contain results that:
- are offensive
- cause misinformation
- are trying to spread hatred
- are not appropriate for children
- are inconsiderate towards a particular group

These results are inappropriate despite having good user engagement. How do we solve this problem? How do we make sure that a search engine can be safely used by users of all age groups and doesn’t spread misinformation and hatred?

#### ML problem
From a machine learning point of view, we would want to have a specialized model that removes inappropriate results from our ranked result set. Hence, we would need training data, features, and a trained classifier for filtering these results.

**Training data**
- **Human raters**: Human raters can identify content that needs to be filtered. We can collect data from raters about the above-mentioned cases of misinformation, hatred, etc. Then, we can train a classifier that predicts the probability that a particular document is inappropriate to show on SERP.
- **Online user feedback**: Another way to generate data is through this kind of online user feedback. This data can then be used to train another model to filter such results.

**Features**
- We can use the same features for this model that we have used for training our ranker, e.g., document word embeddings or raw terms can help us identify the type of content on the document.
- There are maybe a few particular features that we might want to add specifically for our filtering model, such as `website historical report rate, sexually explicit terms used, domain name, website description, images used on the website`, etc.

**Building a classifier**
- We can utilize classification algorithms like logistic regression, MART(Boosted trees or Random Forest), or a Deep neural network to classify a result as inappropriate.
- Similar to the discussion in the ranking section, your choice of the modelling algorithm will depend on:
  - how much data you have
  - capacity requirements
  - experiments to see how much gain in reducing bad content do we see with that modelling technique.

<br>

## Feed Based System
### Problem statement
Design a Twitter feed system that will show the most relevant tweets for a user based on their social graph.

#### Visualizing the problem
- User A is connected to other people/businesses on the Twitter platform. They are interested in knowing the activity of their connections through their feed.
- In the past, a simple approach is displaying all the Tweets generated by their followees since user A’s last visit in reverse chronological order.
- However, this **reverse-chronological order** feed display often resulted in user A missing out on some Tweets that they would have otherwise found very engaging
- Twitter experiences a large number of daily active users, and as a result, the amount of data generated on Twitter is torrential. Therefore, a potentially engaging Tweet may have gotten pushed further down in the feed because a lot of other Tweets were posted after it.
- Hence, to provide a more engaging user experience, it is crucial to **rank the most `relevant` Tweets above the other ones** based on **user `interests`** and **social `connections`**.

#### Scale of the problem
- Let’s define the scope of the problem:
  - Consider that there are 500 million daily active users.
  - On average, every user is connected to 100 users.
  - Every user fetches their feed 10 times in a day.
- 500 million daily active users, each fetching their feed ten times daily, means that your Tweet ranking system will run 5 billion times per day!
- Finally, let’s set up the machine learning problem:
> “Given a list of tweets, train an ML model that predicts the probability of engagement of tweets and orders them based on that score”

<br>

### Metrics
The feed-ranking system aims to **maximize user engagement**. So, let’s start by looking at all the user actions on a Twitter feed.

#### User actions
The following are some of the actions that the user will perform on their tweet, categorized as `positive` and `negative` actions.
- **Positive user actions**
  - Time spent viewing the tweet
  - Liking a Tweet
  - Retweeting
  - Commenting on a Tweet
- **Negative user actions**
  - Hiding a Tweet
  - Reporting Tweets as inappropriate

#### User engagement metrics

<img src="imgs/Twitter_user_actions_report.png" width = "800" height = "450">

<br>

The above illustration shows different `positive` and `negative` engagements on a Twitter feed. Let’s see which of these engagements would be a good one to target as your overall system metric to optimize for.

**Selecting feed optimization metric**
- Selecting a topline is that it’s scientific as well as a business-driven decision.
- The business might want to focus on one aspect of user engagement. 
  - For instance, Twitter can decide that the Twitter community needs to engage more actively in a dialogue. So, the topline metric would be to focus more on `the number of comments` on the Tweets.
  - Similarly, Twitter might want to shift its focus to `overall engagement`. Then their objective will be to increase average overall engagement, i.e., `comments, likes, and retweets`. 
  - Alternatively, the business may require to optimize for `the time spent` on the application. 

**Negative engagement or counter metric**
- For any system, it’s super important to think about counter metrics along with the key, topline ones. 
- In a feed system, users may perform multiple negative actions such as reporting a Tweet as inappropriate, block a user, hide a Tweet, etc.
- Keeping track of these negative actions and having a metric such as `average negative action per user` is also crucial to measure and track.

**Weighted engagement**
- More often than not, all engagement actions are equally important. However, some might become more important at a particular point in time, based on changing business objectives. 
  - For example, to have an engaged audience, `the number of comments` might be more critical rather than just likes. 
- As a result, we might want to have different weights for each action and then track the overall engagement progress based on **the weighted sum**. It will summarize multiple impacts (of different forms of user engagements) into a single score.

<img src="imgs/twitter_value_model.png" width = "600" height = "430">

<br>

- `Normalized Score`: obtain the engagement `per active user`, making the score comparable.
- The weights can be tweaked to find the desired balance of activity on the platform. For example, if we want to increase the focus on `commenting`, we can increase its weight.

<br>

### Architecture

<img src="imgs/feed_based_arch.png" width = "620" height = "260">

<br>

**Tweet selection**
- This component fetches a pool of Tweets from the user’s network (the `followees`), since their last login. This pool of Tweets is then forwarded to the ranker component.

**Training data generation**
- Each user engagement action on the Twitter feed will generate `positive` and `negative` training examples for the user engagement prediction models.

**Ranker**
- The ranker component will receive the pool of selected Tweets and **predict their probability of engagement**. The Tweets will then be ranked according to their predicted engagement probabilities for display on user A’s feed. If we zoom into this component, we can:
  - **Train a single model** to predict the overall engagement on the tweet.
  - **Train separate models**. Each model can focus on predicting the occurrence probability of a certain user action for the tweet. 
    - There will be a separate predictor for `like, comment, time spent, share, hide, and report`. The results of these models can be merged, each having a different weight/importance, to generate a rank score.
- **Separately predicting each user action allows us to have greater control over the importance we want to give to each action** when calculating the rank of the Tweet. 

<img src="imgs/twitter_ranker.png" width = "920" height = "280">

<br>

<br>

### Tweet selection
#### New Tweets
- Fetches 500 newly generated Tweets by user A’s network since the last login
- A not-so-new Tweet!
  - Consider a Tweet that user A has viewed previously. However, by the time the user logs in again, **this Tweet has received a much bigger engagement and/or A’s network has interacted with it**. In this case, it once again becomes relevant to the user due to its recent engagements. 
  - Now, even though this Tweet isn’t new, the Tweet selection component will select it again so that the user can view the recent engagements on this Tweet.

#### New Tweets + unseen Tweets
- The new Tweets fetched in the previous step are ranked and displayed on user A’s feed. They only view the first 200 Tweets and log out. The ranked results are also stored in a cache.
- Now, user A logs in again at 10 pm. According to the last scheme, the Tweet selection component should select all the new Tweets generated between 9 am and 10 pm. However, this may not be the best idea!
- For 9am’s feed, the component previously selected a Tweet made at 8:45 am. **Since this Tweet was recently posted, it did not have much user engagement at that time.** Therefore, it was ranked at the $450^{th}$ position in the feed. Now, remember that A logged out after only viewing the first 200 Tweets and this Tweet remained unread.
- Since the user’s last visit, this unread Tweet gathered a lot of attention in the form of reshares, likes, and comments. The Tweet selection component should now reconsider this Tweet’s selection. This time it would be ranked higher, and A will probably view it.
- Keeping the above rationale in mind, the Tweet selection component now fetches **a mix of newly generated Tweets along with a portion of unseen Tweets from the cache**

**Edge case: User returning after a while**
- If user A log in after two weeks. A lot of Tweets would have been generated since A’s last login. Therefore, the Tweet selection component will **impose a limit on the Tweet data it will select.** 
- Let’s assume that the limit is 500. Now the selected 500 Tweets may have been generated over the last two days or throughout the two weeks, depending on the user’s network’s activity.

#### Network Tweets + interest / popularity-based Tweets
- There could be Tweets outside of user A’s network that have a high potential of engaging them. Hence, we arrive at a two-dimensional scheme of selecting network Tweets and potentially engaging Tweets.
- An engaging Tweet could be one that:
  - aligns with user A’s interests
  - is locally/globally trending
  - engages user A’s network
- Selecting these Tweets can prove to be very beneficial in two cases:
  - The **user has recently joined** the platform and follows only a few others. As such, their small network is not able to generate a sufficient number of Tweets for the Tweet selection component (known as the Bootstrap problem).
  - The user likes a Tweet from a person outside of his network and decides to add them to their network. This would **increase the discoverability on the platform and help grow the user’s network**.

<br>

### Feature Engnieering
Let’s begin by identifying the four main factors in a twitter feed:
- The logged-in user
- The Tweet
- Tweet’s author
- The context
  - Day of week, location, user interests

<img src="imgs/tweet_features.png" width = "900" height = "280">

<br>

#### User-author features
- These features are based on the `logged-in user` and the `Tweet’s author`. They will capture the social relationship between the user and the author of the Tweet, which is an extremely important factor in ranking the author’s Tweets. 
- How can you capture this relationship in your signals given users are not going to specify them explicitly? Following are a few features that will effectively capture this.

**User-author historical interactions**
- When judging the relevance of a Tweet for a `user`, the relationship between the `user` and the Tweet’s `author` plays an important role. It is highly likely that if the `user` has actively engaged with a `followee` in the past, they would be more interested to see a post by that person on their feed.
- Few features based on the above concept can be:
  - `author_liked_posts_3months`: This considers the percentage of an author’s Tweets that are liked by the user in the last three months. 
    - For example, if the author created 12 posts in the last three months and the user interacted with 6 of these posts then the feature’s value will be: `6/12 = 0.5`
    - This feature shows a **more recent trend** in the relationship between the user and the author.
  - `author_liked_posts_count_1year`: This considers the number of an author’s Tweets that the user interacted with, in the last year. 
    - This feature shows a **more long term trend** in the relationship between the `user` and the `author`.
> Ideally, we should normalize the above features by the **total number of Tweets that the user interacted with during these periods**.

#### User-author similarity
Another immensely important feature set to predict user engagement focuses on figuring out how similar the `logged-in user` and the `Tweet’s author` are. A few ways to compute such features include:

**`common_followees`**
- This is a simple feature that can show the similarity between the `user` and the `author`. For a `user-author` pair, you will look at **the number of `users` and `hashtags`** that are followed by both the `user` and the `author`.

**`topic_similarity`**
- The user and author similarity can also be judged based on their interests. You can see if they interact with **similar topics/hashtags**. A simple method to check this is the **bag-of-words or TF-IDF based similarity** between the hashtags:
  - **followed by** the logged-in user and author
  - **present in the posts** that the logged-in user and author have interacted with in the past
  - **used by** the author and logged-in user in their posts
- The similarity between their **search histories** on Twitter can also be used for the same purpose.

**`tweet_content_embedding_similarity`**
- As **`user` is represented by the content that they have generated and interacted with in the past**. You can utilize all of that content as a `bag-of-words` and build an embedding for every user. With an embedding vector for each user, the dot product between them can be used as a fairly good estimate of user-to-author similarity.

**`social_embedding_similarity`**
- Another way to capture the similarity between the user and the author is to generate embeddings based on the **social graph** rather than based on the content of Tweets 
- A basic way to train this model is to represent each `user` with all the other `users` and `topics` (user and topic ids) that they follow in the social graph. 
- Use GNN to get the embeddings

#### Author features

**Author’s degree of influence**

A Tweet written by a more influential author may be more relevant. There are several ways to measure the author’s influence:
- `is_verified`: If an author is verified, it is highly likely that they are somebody important, and they have influence over people.
- `author_social_rank`: The idea of the author’s social_rank is similar to [Google’s page rank](https://en.wikipedia.org/wiki/PageRank#Algorithm).
  - To compute the social rank of each user, you can do a **random walk** (like in page rank). Each person who follows the author contributes to their rank. 
  - However, the contribution of each user is not equal. For example, **a user adds more weight if they are followed by a popular celebrity or a verified user.**
- `author_num_followers`: One important feature could be the number of followers the author has. Different inferences can be drawn from different follower counts, as shown below:
  - `Normalised author_num_followers`: `min_max` normalization
  - bucketing into several bins

| Follower count |                              Inference                              |
| :------------: | :-----------------------------------------------------------------: |
|      150       | Personal account with reasonable influence over their social circle |
|   1 thousand   |  Social media influencer with a good amount of influence over fans  |
|   3 million    |                Celebrity with a great fan following                 |

**Historical trend of interactions on the author’s Tweets**
Another very important set of features is the interaction history on the author’s Tweets. If historically, an author’s Tweets garnered a lot of attention, then it is highly probable that this will happen in the future, too.

> A high rate of historical interaction for a user implies that the user posts high-quality content.

Some features to capture the historical trend of interactions for the author’s Tweets are as follows:
- `author_engagement_rate_3months`: To compute the engagement rate in the last three months, we look at how many times different users interacted with the author’s Tweets that they viewed.
$$
\text{Engagement rate} = \frac{\text{tweet interactions}}{\text{tweet views}}
$$
- `author_topic_engagement_rate_3months`: The engagement rate can be different based on the Tweet’s topic. 
  - For example, if the author is a sports celebrity, the engagement rate for their `family-related Tweets` may be different from the engagement rate for their `sports-related Tweets`.
  - We can capture this difference by computing the engagement rate for the author’s Tweets per topic. Tweet topics can be identified in the following two ways:
    - Deduce the Tweet topic by the hashtags used
    - Predict the Tweet topic based on its content

#### User-Tweet features
The similarity between the user’s interests and the tweet’s topic is also a good indicator of the relevance of a Tweet. For instance, if user A is interested in `football`, a Tweet regarding `football` would be very relevant to them.
- `topic_similarity`: You can use the `hashtags` and/or the `content` of the Tweets that the user has either Tweeted or interacted with, in the last 6 months and compute the `TF-IDF similarity` with the Tweet itself
- `embedding_similarity`: Another option to find the similarity between the user’s interest and the Tweet’s content is to generate embeddings for the user and the Tweet. 
  - The Tweet’s embedding can be made based on the content and hashtags in it. 
  - The user’s embedding can be made based on the content and hashtags in the Tweets that they have written or interacted with. 

#### Tweet features

##### Features based on Tweet’s content
- `Tweet_length`: 
  - The length of the Tweet positively correlates with user engagement, especially likes and reshares.
  - It is generally observed that **people have a short attention span and prefer a shorter read**. Hence, a more concise Tweet generally increases the chance of getting a like by the user. 
- `Tweet_recency`
  - The recency of the Tweet is an important factor in determining user engagement, as people are most interested in the latest developments.
- `is_image_video`
  - The presence of an image or video makes the tweet more catchy and increases the chances of user engagement.
- `is_URL`
  - The presence of a URL may mean that the Tweet:
    - Calls for action
    - Provides valuable information
  - Hence, such a Tweet might have a higher probability of user engagement.

##### Features based on Tweet’s interaction
Tweets with a greater volume of interaction have a higher probability of engaging the user.
- `num_total_interactions`: The total number of interactions (likes, comments and reshares) on a Tweet can be used as a feature.

**Caveat**: Simply using these interactions as features might give an incomplete picture without considering time factor. 
- To remedy this, we can apply a simple **time decay model** to weight the latest interaction more than the ones that happened some time ago.

> Time decay can be used in all features where there is a decline in the value of a quantity over time.
- Another remedy is to use different **time windows** to capture the recency of interactions while looking at their numbers. The interaction in each window can be used as a feature:
  - `interactions_in_last_1_hour`
  - `interactions_in_last_1_day`
  - `interactions_in_last_3_days`
  - `interactions_in_last_week`

##### Separate features for different engagements
Previously, we discussed combining all interactions. You can also keep them as separate features, given you can predict different events, e.g., the probability of likes, Retweets and comments. Some potential features can be:
- `likes_in_last_3_days`
- `comments_in_last_1_day`
- `reshares_in_last_2_hours`

Another set of features can be generated by looking at the interactions on the Tweet made only by user A’s network. The intuition behind doing this is that there is a high probability that if a Tweet has more interactions from A’s network, then A, having similar tastes, will also interact with that Tweet. The set of features based on user’s network’s interactions would then be:
- `likes_in_last_3_days_user’s_network_only`
- `comments_in_last_1_day_user’s_network_only`
- `reshares_in_last_2_hours_user’s_network_only`

#### Context-based features
- `day_of_week`
- `time_of_day`
  - Noting the time of the day (coupled with the day of the week) can provide useful information. 
  - For example, if a user logs-in on a Monday morning, it is likely that they are at their place of work. Now, it would make sense to show shorter Tweets that they can quickly read. In contrast, if they login in the evening, they are probably at their home and would have more time at their disposal.
- `current_user_location`
  - The current user location can help us show relevant content. For example, a person may go to San Francisco where a festival is happening (it is the talk of the town). Based on what is popular in a certain area where the user is located, you can show relevant Tweets.
- `season`
- `lastest_k_tag_interactions`
- `approaching_holiday`

#### Sparse features
- `unigrams/bigrams of a Tweet`
- `user_id`
- `tweets_id`
- use embeddings to get dense representation

<br>

### Training Data Generation
#### Data generation through online user engagement
- If you are training a **single model** to predict user engagement,
  - all the Tweets that received user engagement would be labeled as `positive` training examples.
  - all the Tweets that only have impressions would be labeled as `negative` training examples.
- When you generate data for the **“Like” prediction model**, 
  - all Tweets that the user **has liked** would be `positive` examples, 
  - all the Tweets that they **did not like** would be negative examples.
    - Note how the `comment` is still a `negative` example for the “Like” prediction model.

#### Balancing positive and negative training examples
- In the feed-based system scenario, on average, a user engages with as little as approximately 5% of the Tweets that they view per day. How would this percentage affect the ratio of `positive` and `negative` examples on a larger scale? How can this be balanced?
- Looking at the bigger picture, assume that 100 million Tweets are viewed collectively by the users in a day. With the 5% engagement ratio, you would be able to generate 5 million positive examples, whereas the remaining 95 million would be negative.
- Let’s assume that you want to limit your training samples to 10 million, given that the training time and cost increases as training data volume increases. In order to balance the ratio of `positive` and `negative` training samples, you can randomly **downsample**:
  - negative examples to 5 million samples
  - positive examples to 5 million samples
- Now, you would have a total of 10 million training examples per day
- **Note**: Due to downsampling, we have changed the sampling of training data, our model output scores will **not be well-calibrated**. However, because we only need to rank Tweets, **poor model calibration doesn’t matter much in this scenario**
  - If a model is well-calibrated, the distribution of its predicted probability is similar to the distribution of probability observed in the training data

<br>

### Ranking
The task is to predict the probability of different engagement actions for a given Tweet. So, essentially, your models are going to predict the probability of `likes, comments, retweets`, i.e., `P(click), P(comments), P(retweet)`

#### Logistic regression
- Initially, a simple model that makes sense to train is a logistic regression model with regularization, to predict engagement using the **dense features**.
  - logistic regression is that it is reasonably fast to train. This enables you to test new features fairly quickly to see if they make an impact on the AUC or validation error. 
  - Also, it’s extremely easy to understand the model and feature importance
- A major **limitation** of the linear model is that it assumes linearity exists between the input features and prediction. Therefore, you have to **manually model feature interactions**. 
  - For example, if you believe that the day of the week before a major holiday will have a major impact on your engagement prediction, you will have to create this feature in your training data manually.
- Another key question is whether you want to train a single classifier for overall engagement or separate ones for each engagement action based on production needs. 
  - In a **single classifier** case, you can train a logistic regression model for predicting the overall engagement on a Tweet. Tweets with any form of engagement will be considered as positive training examples for this approach.
  - Another approach is to **train separate logistic regression models** to predict P(like), P(comments) and P(retweet).

#### MART
- MART: Multiple Additive Regression Trees.
- Another modeling option that should be able to outperform logistic regression **with dense features** is additive tree-based models, e.g., `GBDT` and `Random Forest`. 
  - Trees are inherently able to utilize non-linear relations between features that aren’t readily available to logistic regression
- There are various hyperparameters that you might want to play around to get to an optimized model, including
  - `Number of trees`
  - `Maximum depth`
  - `Minimum samples needed for split`
  - `Maximum features sampled for a split`
- Option 1: train a single model to predict the overall engagement.
- Option 2: train several models to predict different kinds of engagement
- Option 3: build one common model, i.e., P(engagement) and share its output as input into all the seperate models.

#### Deep learning
- Use multi-layer approach: having a simpler model for stage 1 ranking and use complex stage 2 model to obtain the most relevant Tweets ranked at the top of the user’s Twitter feed.
- A few hyperparameters to tune:
  - Learning rate
  - Number of hidden layers, number or neurons
  - Batch size
  - Number of epochs
  - Dropout rate, weight decay for regularizing model and avoiding overfitting

##### Multi-task neural networks
- Since predicting `P(like), P(comment), P (retweet)` are similar tasks. When predicting similar tasks, you can use multitask learning.
- Hence, you can train a neural network with shared layers (for shared knowledge) appended with specialized layers for each task’s prediction. 
  - The weights of the shared layers are common to the three tasks. Whereas in the task-specific layer, the network learns information specific to the tasks. 
  - The loss function for this model will be the sum of individual losses for all the tasks:

`total_loss = like_loss + comment_loss + retweet_loss`

- Classic Multi-task learning frameworks:
  - MoME

#### Stacking models and online learning

<br>

### Diversity
#### Diversity in Tweets’ authors
- It is possible that the sorted list of Tweets has 5 consecutive posts by the same author! No matter how good of a friend the author is or how interesting their Tweets might be, user A would eventually get bored of seeing Tweets from the same author repeatedly. Hence, you need to introduce diversity with regards to the Tweets’ author.

#### Diversity in tweets’ content
- Another scenario where we might need to introduce diversity is the Tweet’s content. For instance, if your sorted list of Tweets has 4 consecutive tweets that have videos in them, the user might feel that their feed has too many videos.

#### Introducing the repetition penalty
- To rid the Twitter feed from a monotonous and repetitive outlook, we will introduce a repetition penalty for **repeated Tweet authors and media content in the Tweet**.
- One way to introduce a repetition penalty could be to **add a negative weight** to the Tweet’s score upon repetition. 
  - For instance, whenever you see the author being repeated, you add a negative weight of `-0.1` to the Tweet’s score.
- Another way to achieve the same effect is to **bring the Tweet with repetition 3 steps down in the sorted list**. 
  - For instance, when you observe that two consecutive Tweets have media content in them, you bring the latter down by 3 steps.

<br>

### Online Experimentation
- **Step 1: Training different models** 
  - different features sets
  - different model options
  - different hyper-params
- **Step 2: Validating models offline**
  - AUC on unseen test-set
- **Step 3: Online experimentation**
  - baseline/control group: displays the feed in reverse chronological order
  - variation group: best model selected at Step 2
  - select 1% of the 500 million active users, i.e., 5 million users for the A/B test. Two buckets of these users will be created each having 2.5 million users. 
    - Bucket 1 users will be shown twitter timelines according to the time-based model -- control group
    - Bucket 2 users will be shown the Twitter timeline according to the new ranking model.
  - However, before you perform this A/B test, you need to **retrain the ranking model**.
    - Recall that we withheld the most recent partition of the training data to use for validation and testing. This was done to check if the model would be able to predict future engagements on tweets given the historical data. 
    - However, now that you have performed the validation and testing, you need to retrain the model using the recent(or all) partitions of training data so that it captures the most recent phenomena.
- **Step 4: To deploy or not to deploy**
  - Use statistical significance (like p-value) to ensure that the gain of the ranking model is real.
  - Another aspect to consider when deciding to launch the model on production, especially for **smaller gains**, is the increase in complexity. If the new model increases the complexity of the system significantly without any significant gains, you should not deploy it.

<br>

## Recommendation System
### Problem statement
The interviewer has asked you to display media (movie/show) recommendations for a Netflix user. Your task is to make recommendations in such a manner that the chance of the user watching them is maximized.

#### Scope of the problem
Let’s define the scope of the problem:
- The total number of subscribers on the platform as of 2019 is 163.5 million.
- There are 53 million international daily active users.

One common way to set up the recommendation system in the machine learning world is to pose it as a classification problem with the aim to predict the probability of user engagement with the content. So, the problem statement would be:

> “Given a user and context (time, location, and season), predict the probability of engagement for each movie and order movies using that score.”

#### Problem formulation
We established that you will predict the probability of engagement for each movie/show and then order/rank movies based on that score. The recommendation system would be based on **implicit feedback** (having binary values: the user has watched movie/show or not watched).

Let’s see **why we have opted for ranking movies using “implicit feedback”** as our probability predictor instead of using “explicit feedback” to predict movie ratings.

**Types of user feedback**

Generally, there are two types of feedback coming from an end-user for a given recommendation.
- **Explicit feedback**: A user provides an explicit assessment of a recommendation. In our case, it would be a star rating
  - Here, the recommendation problem will be viewed as a **rating prediction problem**.
- **Implicit feedback**: Implicit feedback is extracted from a user’s interaction with the recommended media. Most often, it is **binary** in nature. For instance, a user watched a movie (1), or they did not watch the movie (0).
  - Here, the recommendation problem will be viewed as a **ranking problem**.

One key advantage of utilizing implicit feedback is that it allows collecting a large amount of training data. 

**Explicit feedback** faces the **missing not at random (MNAR)** problem. 
- Users will generally rate those media recommendations that they liked. This means `4/5, 5/5` star ratings are more common than `1/5, 2/5`. 
- Therefore, we won’t get much information on the kind of movies that a user does not like. 
- Also, movies with fewer ratings would have less impact on the recommendation process.

<br>

### Metrics
#### Online metrics
**Engagement rate**
- The success of the recommendation system is directly proportional to the number of recommendations that the user engages with. So, the engagement rate $\left(\frac{\text { sessions with clicks }}{\text { total number of sessions }}\right)$ can help us measure it. 
- However, the user might click on a recommended movie but does not find it interesting enough to complete watching it. Therefore, only measuring the engagement rate with the recommendations provides an incomplete picture.

**Videos watched**
- To take into account the unsuccessful clicks on the movie/show recommendations, we can also consider **the average number of videos that the user has watched**. We should **only count videos that the user has spent at least a significant time watching** (e.g., more than 2 minutes).
- However, this metric can be problematic when it comes to the user starting to watch movie/series recommendations but not finding them interesting enough to finish them.
  - Series generally have several `seasons` and `episodes`, so watching one episode and then not continuing is also an indication of the user not finding the content interesting. 

**Session watch time**
- Session watch time measures **the overall time a user spends watching content based on recommendations in a session**. The key measurement aspect here is that the user is able to find a meaningful recommendation in a session such that they spend significant time watching it.
- To illustrate intuitively on why session watch time is a better metric than **engagement rate** and **videos watched**, let’s consider an example of two users, `A` and `B`. 
  - User `A` engages with 5 recommendations, spends 10 minutes watching 3 of them and then ends the session. 
  - One the other end, user `B` engages with 2 recommendations, spends 5 minutes on 1 and then 90 minutes on the second recommendation. 
  - Although user `A` engaged with more content, user `B`’s session is clearly more successful as they found something interesting to watch.

#### Offline metrics
##### `mAP@N`
- `mAP@N`: Mean Average Precision, N = length of the recommendation list
$$
\mathrm{Precision}=\frac{\text { number of relevant recommendations }}{\text { total number of recommendations }}
$$
- We can observe that `precision` alone **does not reward the early placement of relevant items** on the list. However, if we calculate the precision of the subset of recommendations up until each position, $\mathbf{k}(\mathrm{k}=1$ to $\mathrm{N}$ ), on the list and take their weighted average, we will achieve our goal. 
- Assume the following:
  - The system recommended $N = 5$ movies.
  - The user watched 3 movies from this recommendation list and ignored the other 2.
  - Among all the possible movies that the system could have recommended (available on the Netflix platform), only $m = 10$ are actually **relevant to the user** (historical data).

<img src="imgs/AP@N.png" width = "930" height = "300">

<br>

- Now to calculate the average precision (AP), we have the following formula:
$$
\text { AP@N }=\frac{1}{m} \sum_{k=1}^{N}\left(P(k) \text { if } k^{t h} \text { item was relevant }\right)=\frac{1}{m} \sum_{k=1}^{N} P(k) \cdot \operatorname{rel}(k)
$$
  - In the above formula, $\operatorname{rel}(k)$ tells whether that $k^{th}$ item is relevant or not.
  - Here, we see that $P(k)$ only contributes to $\text{AP}$ if the recommendation at position $k$ is relevant.
  - Lastly, the “mean” in `mAP` means that we will **calculate the `AP` with respect to each user’s ratings and take their mean**. So, **`mAP` computes the metric for a large set of users** to see how the system performs overall on a large set.

##### `mAR@N`
- `mAR@N`: Mean Average Recall, N = length of the recommendation list
$$
\mathrm{Recall}=\frac{\text { number of relevant recommendations }}{\text { number of all possible relevant items }}
$$
- We will use the same recommendation list as used in the `mAP@N` example, where $\mathrm{N}=5$ and $\mathrm{m}=10$. 
- The **Average Recall (AR)** will then be calculated as follows:

<img src="imgs/AR@N.png" width = "480" height = "260">

<br>

- Lastly, the “mean” in `mAR` means that we will **calculate `AR` with respect to each user’s ratings and then take their mean**.
- So, `mAR` at a high-level, measures how many of the top recommendations (based on historical data) we are able to get in the recommendation set.

##### `F1 score`
$$
\text{F1 score} = 2 \times \frac{mAR \times mAP}{mAP + mAR}
$$
 Remember that we selected our recommendation set size to be 5, but it can be differ based on the recommendation viewport or the number of recommendations that users on the platform generally engage with.

##### `RMSE`
We established above that we optimize the system for implicit feedback data. However, what if the interviewer says that you have to optimize the recommendation system for getting the **ratings (explicit feedback)**. Here, it makes sense to use root mean squared error (RMSE) to minimize the error in rating prediction.
$$
\mathrm{RMSE}=\sqrt{\frac{1}{N} \sum_{i=1}^{N}\left(\hat{y}_{i}-y_{i}\right)^{2}}
$$
<br>

### Architectural Components
Since we We have a huge number of movies to choose from, we split the recommendation task into two stages.
- **Stage 1: Candidate generation**: uses a simpler mechanism to sift through the entire corpus for possible recommendations
- **Stage 2: Ranking of generated candidates**: uses complex strategies only on the candidates given by stage 1 to come up with personalized recommendations

<img src="imgs/recommendation_arch.png" width = "730" height = "400">

<br>

#### Candidate generation
This component uses several techniques to find out the best candidate movies/shows for a user, given the user’s historical interactions with the media and context.
> This component focuses on **higher recall**, meaning it focuses on gathering movies that might interest the user from all perspectives, e.g., media that is relevant based on historical user interests, trending locally, etc.

#### Ranker
The ranker component will score the candidate movies/shows generated by the candidate data generation component according to how interesting they might be for the user.
> This component focuses on **higher precision**, i.e., it will focus on the ranking of the top k recommendations.

It will ensemble different scores given to a media by multiple candidate generation sources whose scores are not directly comparable. 

#### Training data generation
The user’s engagement with the recommendations on their Netflix homepage will help to generate training data for both the candidate generation component and the ranker component

<br>

### Feature Engineering
To start the feature engineering process, we will first identify the main factors in the movie/show recommendation process:
- Logged-in user
- Movie/show
- Context (e.g., season, time, etc.)

The features would fall into the following categories:
- **User-based** features
- **Context-based** features
- **Media-based** features
- **Media-user** cross features

A subset of the features is shown below.

<img src="imgs/recommendation_features.png" width = "930" height = "200">

<br>

#### User-based features
- optional: due to bias issue
  - **`age`**: This feature will allow the model to learn the kind of content that is appropriate for different age groups and recommend media accordingly.
  - **`gender`**: The model will learn about gender-based preferences and recommend media accordingly.
- **`language`**: This feature will record the language of the user. It may be used by the model to see if a movie is in the same language that the user speaks.
- **`country`**: This feature will record the country of the user. Users from different geographical regions have different content preferences. This feature can help the model learn geographic preferences and tune recommendations accordingly.
- **`average_session_time`**: This feature (user’s average session time) can tell **whether the user likes to watch lengthy or short movies/shows**.
- **`last_genre_watched`**: The genre of the last movie that a user has watched may serve as a hint for what they might like to watch next.

The following are some user-based features (derived from historical interaction patterns) that have a **sparse** representation.

- **user_actor_histogram**: This feature would be a **vector based** on the histogram that shows the historical interaction between the active user and all actors in the media on Netflix. It will record the **percentage of media that the user watched with each `actor` cast in it**.
- **`user_genre_histogram`**: This feature would be a **vector based** on the histogram that shows historical interaction between the active user and all the genres present on Netflix. It will record the **percentage of media that the user watched belonging to each `genre`**.
- **`user_language_histogram`**: This feature would be a **vector based** on the histogram that shows historical interaction between the active user and all the languages in the media on Netflix. It will record **the percentage of media in each `language` that the user watched**.

#### Context-based features
Making context-aware recommendations can improve the user’s experience

- **`season_of_the_year`**: User preferences may be patterned according to the four seasons of the year. This feature will record the season during which a person watched the media. 
- **`upcoming_holiday`**: This feature will record the upcoming holiday. People tend to watch holiday-themed content as the different holidays approach. 
  - Holidays will be region-specific as well.
- **`days_to_upcoming_holiday`**: It is useful to see how many days before a holiday the users started watching holiday-themed content. 
- **`time_of_day`**: A user might watch different content based on the time of the day as well.
- **`day_of_week`**: User watch patterns also tend to vary along the week. 
- **`device`**: It can be beneficial to observe the device on which the person is viewing content. 
  - A potential observation could be that users tend to watch content for shorter periods on their mobile when they are busy. 
  - They usually chose to watch on their TV when they have more free time. So, they watch media for a longer period consecutively on their TV. 

#### Media-based features
We can create a lot of useful features from the media’s metadata.

- **`public_platform_rating`**: This feature would tell the public’s opinion, such as IMDb/rotten tomatoes rating, on a movie. 
- **`revenue`**: We can also add the revenue generated by a movie before it came to Netflix. This feature also helps the model to figure out the movie’s popularity.
- **`time_passed_since_release_date`**: The feature will tell how much time has elapsed since the movie’s release date.
- **`time_on_platform`**: It is also beneficial to record how long a media has been present on Netflix.
- **`media_watch_history`**: Media’s watch history (number of times the media was watched) can indicate its popularity. 
  - Some users might like to stay on top of trends and focus on only watching popular movies. They can be recommended popular media. 
  - Others might like less discovered indie movies more. They can be recommended less watched movies that had good implicit feedback (the user watched the whole movie and did not leave it midway).
  - We can look at the media’s watch history for different time intervals as well:
    - **`media_watch_history_last_12_hrs`**
    - **`media_watch_history_last_24_hrs`**
- **`genre`**: This feature records the primary genre of content, e.g., comedy, action, documentaries, classics, drama, animated, and so on.
- **`movie_duration`**: This feature tells the movie duration.
- **`content_set_time_period`**: This feature describes the time period in which the movie/show was set in. For example, it may show that the user prefers shows that are set in the '90s.
- **`content_tags`**: Netflix has hired people to watch movies and shows to create extremely detailed, descriptive, and specific tags for the movies/shows that capture the nuances in the content. 
  - For instance, media can be tagged as a “Visually-striking nostalgic movie”. These tags greatly help the model understand the taste of different users and find the similarity between the user’s taste and the movies.
- **`show_season_number`**: If the media is a show with multiple seasons, this feature can tell the model whether a user likes shows with fewer seasons or more.
- **`country_of_origin`**: This feature holds the country in which the content was produced.
- **`release_country`**: This feature holds the country where the content was released.
- **`release_year`**: This feature shows the year of theatrical release, original broadcast date or DVD release date.
  - compute the difference against current year?
- **`maturity_rating`**: This feature contains the maturity rating of the media with respect to the territory (geographical region). 

#### Media-user cross features
In order to learn the users’ preferences, representing their **historical interactions with media** as features is very important. Some of these interaction-based features are as follows:

**User-genre historical interaction features**

These features represent the percentage of movies that the user watched with the same genre as the movie under consideration. This percentage is calculated for different time intervals to cater to the dynamic nature of user preferences.

- **`user_genre_historical_interaction_3months`**: The percentage of movies that the user watched with the same genre as the movie under consideration in the last 3 months. 
  - For example, if the user watched 6 comedy movies out of the 12 he/she watched in the last 3 months, then the feature value will be `0.5`
- **`user_genre_historical_interaction_1year`**
- **`user_and_movie_embedding_similarity`**
  - Netflix has hired people to watch movies and shows to create incredibly detailed, descriptive, and specific tags for the movies/shows that capture the nuances in the content. For instance, media can be tagged as “Visually-striking nostalgic movie”.
  - You can have **a user embedding based on the `tags` of movies** that the user has interacted with and a media embedding based on its tags.
- **`user_actor`**: This feature tells the percentage of media that the user has watched, which has the same cast (actors) as that of the media under consideration for recommendation.
- `user_director`
- `user_language_match`: This feature matches the user’s language and the media’s language.

**Sparse features**
- `movie_id`: Popular movie IDs are be repeated frequently.
- `title_of_media`
- `synopsis`: This feature holds the synopsis or summary of the content.
- `original_title`: This feature holds the original title of the movie in its original language. 
- `distributor`: A particular distributor may be selecting very good quality content, and hence users might prefer content from that distributor.
- `creator`
- `director`
- `music_composer`
- `actors`: This feature includes the cast of the movie/show.
  - how to represent a list of actors?
- `original_language`: This feature holds the original spoken language of the content. If multiple, you can record the choose the majority language.
- `first_release_year`

<br>

### Candidate Generation
The candidate generation techniques are as follows:
- **Collaborative filtering**
  - Nearest neighborhood
  - Matrix factorization
- **Content-based filtering**
- **Embedding-based similarity**

Each method has its own strengths for selecting good candidates, and we will **combine all of them together to generate a complete list** before passing it on to the ranked

#### Collaborative filtering
We can find users similar to the **active user** based on the intersection of their historical watches. Then, **collaborate with similar users** to generate candidate media for **the active user**
- Intuitioon: If a user shares a similar taste with a group of users over a subset of movies, they would probably have similar opinions on other movies compared to the opinion of a randomly selected person.

##### Nearest Neighborhood
**User-based** approach of collaborative filtering (CF): identify similar users, use feedbacks of top K similar users about unseen media to get current user $i$'s feedback about the unseen media

<img src="imgs/user_media_matrix.png" width = "350" height = "350">

<br>

- To generate recommendations for user $i$, you need to predict their feedback for all the movies they haven't watched. 
- You will collaborate with users similar to user $i$ for this process. Their ratings for a movie, not seen by user $i$, would give us a good idea of how user $i$ would like it.
- You will compute the similarity (e.g. cosine similarity) of other users with user $i$ and then select the top $\mathrm{k}$ similar users/nearest neighbours $\left(\operatorname{KNN}\left(u_{i}\right)\right)$. 
- Then, user $i$ 's feedback for an unseen movie $j\left(f_{i j}\right)$ can be predicted by taking the **weighted average of feedback** that the top $\mathrm{k}$ similar users gave to movie $j$.
$$
f_{i j}=\frac{\sum_{v \in KNN\left(u_{i}\right)} \operatorname{Similarity}\left(u_{i}, u_{v}\right) f_{v j}}{k}
$$
- The unseen movies with good predicted score will be chosen as candidates for user $i$’s recommendations.

**Item-based** approach: we look at the similarity of the items(movies/shows) for generating candidates.
- First, you calculate **media similarity** based on similar feedback from users. 
- Then **the `media` most similar to that already watched by `user A`** will be selected as candidates for user A’s recommendation.

**NOTE**: 
- The **user-based** approach is often **harder to scale** as **`user` preference tends to change over time**. `Items`, in contrast, don’t change so the **item-based** approach can usually be **computed offline** and served without frequent re-training.
- It is evident that the **Nearest Neighborhood** is **computationally expensive** with the increase in numbers of `users` and `movies`. **The sparsity of this matrix** also poses a problem when a `movie` has not been rated by any `user` or a new `user` has not watched many `movies`.

##### Matrix factorization
To represent the `user-media` interaction matrix in a way that is scalable and handles sparsity, matrix factorization helps us by **factoring this matrix into two lower dimensional matrices**:
- User profile matrix (n x D): Each `user` in the user profile matrix is represented by a `row`, which is a latent vector of D dimensions.
- Media profile matrix (D x m): Each `movie` in the movie profile matrix is represented by a `column`, which is a latent vector of D dimensions.

#### Content-based filtering
Content-based filtering allows us to make recommendations to `users` based on the characteristics or attributes of the `media` they have already interacted with.
- The characteristics come from **metadata** (e.g., `genre, movie cast, synopsis, director`, etc.) information and manually assigned **`media_descriptive_tags`** (e.g., `visually striking, nostalgic, magical creatures, character development, winter season, quirky indie rom-com set in Oregon`, etc.) by Netflix taggers. 
- The media is represented as **a vector of its attributes**. The following explanation takes a subset of the attributes for ease of understanding.

<img src="imgs/media_attr_vector.png" width = "700" height = "450">

<br>

<br>

- Initially, you have the media’s attributes in raw form. They need to be preprocessed accordingly to extract features
  - remove stop words
  - convert attribute values to lowercase to avoid duplication. 
  - join the `director’s` first name and last name to identify unique people, similar preprocessing is required for the `tags`. 

<img src="imgs/media_attr_tf-idf_vector.png" width = "800" height = "500">

<br>

<br>

- Now, you have the movies represented as vectors containing the `TF (term/attribute frequency)` of all the attributes. 
  - This is followed by **normalizing the `TF (term frequencies)` by the length of the vectors**. 
  - Finally, you multiply the movies’ normalized `TF` vectors and the `IDF` vector element-wise to make `TF-IDF` vectors for the movies.

Given the `TF-IDF` representation of each movie/show, you have 2 options for recommending media to the user:
- **Similarity with historical interactions**
  - You can recommend `movies` to the `user` similar to those they have interacted (seen) with in the past. This can be achieved by **computing the dot product of `movies`**.
  - **Simple and intuitive**
- **Similarity between `media` and `user` profiles**
  - The media’s TF-IDF vectors can be viewed as their profile. Based on user preferences, which can be seen from its historical interactions with media, you can build user profiles as well as shown below.

<img src="imgs/create_user_profile.png" width = "860" height = "260">

<br>

  - Now, instead of using past watches to recommend new ones, you can just **compute the similarity (dot product) between the `user’s` profile and media profiles of `unseen movies`** to generate relevant candidates for user A’s recommendations.

<img src="imgs/choose_candidates_on_user_profile.png" width = "780" height = "200">

<br>

#### **Generate embedding using neural networks
- Given the historical feedback `(u, m)`, i.e., `user u`’s feedback for `movie m`, you can use the power of deep learning to generate latent vectors/embeddings to represent both `movies` and `users`. Once you have generated the vectors, you will utilize KNN to find the movies that you would want to recommend to the user. 
- Refer to **Youtube Two Tower framework**

**Embedding generation**
- You set up the network as two towers with one tower feeding in `media-only` sparse and dense features and the other tower feeding in `user-only` sparse and dense features. 
- The activation of the first tower’s last layer will form the **media’s vector embedding ($m$)**. Similarly, the activation of the second tower’s last layer will form the **user’s vector embedding ($u$)**. 
- The combined optimization function at the top aims to minimize the distance between the dot product of $u$ and $m$ (predicted feedback) and the actual feedback label.

**Candidate selection (KNN)**
- After the user and media vector embeddings have been generated, you will apply KNN and select candidates for each user.
- Refer to **FAISS**

#### Techniques’ strengths and weaknesses#
- **Collaborative filtering**
  - can suggest candidates based solely on the historical interaction of the users. 
  - Unlike **content-based filtering**, it does not require domain knowledge to create user and media profiles. It may also be able to capture data aspects that are often elusive and difficult to profile using content-based filtering. 
  - However, **collaborative filtering** suffers from the **cold start problem**. 
    - It is difficult to find users similar to a **new user** in the system because they have less historical interaction. 
    - Also, **new media** can’t be recommended immediately as no users have given feedback on it.
- **The neural network(embeddings) technique**
  - also suffers from the **cold start** problem.
    - if a movie is new or if a user is new, both would have fewer instances of feedback received and feedback given, respectively
    - by extension, this means there is a lack of sufficient training examples to update their embedding vectors accordingly.
- **Content-based filtering**
  - can handle **cold start** problem
  - However, it does require some initial input from the user regarding their preferences to start generating candidates. 
    - This input is obtained as a part of the onboarding process, where a **new user** is asked to share their preferences. 
    - Moreover, **new medias**’ profiles can be built immediately as their description is provided manually. 

<br>

### Training Data Generation
#### Generating training examples
One way of interpreting user actions as `positive` and `negative` training examples is based on **the duration** for which the user watched a particular show/movie. 
- `positive` examples: user watched most of a recommended movie/show, i.e., watched 80% or more.
- `negative` examples: user ignored a movie/show or watched 10% or less.
- If the percentage of a movie/show watched by the user falls between 10% and 80%, you will put it in the `uncertainty bucket` and ignore such examples.
  - to avoid misinterpretations, you label examples as **`positive` and `negative` only** when you are certain about it to a higher degree.

#### Balancing positive and negative training examples
You have a lot more `negative` training examples than `positive` ones. To balance the ratio of `positive` and `negative` training samples, you can randomly downsample the `negative` examples.

#### Weighting training examples
Based on our training data discussion so far, all of the training examples are weighted equally. According to Netflix’s business objectives, the **main goal could be to increase the time a user spends on the platform**.

One way to incentivize your model to focus more on examples that have a higher contribution to the session watch time is to **weight examples based on their contribution to session time**. Here, you are assuming that your prediction model’s optimization function utilizes weight per example in its objective.

One **caveat** of utilizing these weights is that the model might only recommend content with a longer watch time. So, it’s important to choose weights such that we are **not solely focused on watch time**. We should find the right balance between user satisfaction and watch time, based on our online A/B experiments.

<br>

### Ranking
The ranking model takes the top candidates from multiple sources of candidate generation and ranks them with respect to the chance of the user watching that video content.

Here, your goal is to rank the content based on the probability of a user watching a media given a user and a candidate media, i.e., `P(watch|(User, Media))`

#### Logistic regression or random forest
There are multiple reasons that training a simplistic model first:
- Training data is limited
- Limited training and model evaluation capacity
- Need model explainability to really understand how the ML model is making its decision and show that to the stake holders
- Need an initial baseline model

**Features used in the model**:
- Dense features created in feature engineering stage
  - User-based features
  - Context-based features
  - Media-based features
  - Media-user cross features

<img src="imgs/recommendation_features.png" width = "930" height = "200">

<br>

- Content-based filtering score for `(user, media)` pair
- Collaborative filtering score for `(user, media)` pair
- Embedding-based filtering score for `(user, media)` pair

#### Deep NN
We can train a deep NN with `sparse` and `dense` features. **Two extremely powerful `sparse` features** fed into such a network can be:
- videos that the user has previously watched
- the user’s search terms. 

For these sparse features, you can set up the network to also learn `media` and `search term` embeddings as part of the learning task. 
- These specialized embeddings for historical watches and search terms can be very powerful in predicting the next watch media for a user. 
- They will allow the model to personalize the recommendation ranking based on the user’s recent interaction with media content on the platform.

An important aspect here is that both `search terms` and `historical watched content` are **list-wise features**. You need to think about how to feed them in the network
- simple pooling strategies:
  - average the historical watch id and search text term embeddings
  - max pooling 
- CNN + pooling
- RNN: last hidden state
  - add attention
- Transformer

**Normal architecture: Wide & Deep**
- Wide side: extreme important features
  - Content-based filtering score for `(user, media)` pair
  - Collaborative filtering score for `(user, media)` pair
  - Embedding-based filtering score for `(user, media)` pair
- Deep side: all the features, embeddings created


#### Re-ranking
Re-ranking is done for various reasons, such as **bringing diversity** to the recommendations. 
- Consider a scenario where all the top ten recommended movies are comedy. You might decide to keep only two of each genre in the top ten recommendations. This way, you would have five different genres for the user in the top recommendations.

If you are also considering past watches for the media recommendations, then re-ranking can help you. It prevents the recommendation list from being overwhelmed by previous watches by **moving some previously watched media down the list of recommendations**.

<br>

## Ad Prediction System
### Problem Statement
The interviewer can ask the following questions about this problem, narrowing the scope of the question each time.
- How would you build an ML system to **predict the probability of engagement for Ads**?
- How would you build an Ads relevance system for a search engine?
- How would you build an Ads relevance system for a social network?

Note that **the context can be different** depending on the type of application in which we are displaying the advertisement. There are two categories of applications:
- `Search engine`: Here, the `query` will be part of the context along with the `searcher`. The system will display ads based on the search `query` issued by the `searcher`.
- `Social media platforms`: Here, we do not have a query, but the **user information** (such as location, demographics, and interests hierarchy) will be part of the context and used to select ads.
  - The system will automatically detect user interest based on the user’s historical interactions and display ads accordingly.

Let’s set up the machine learning problem:

> “Predict the probability of engagement of an ad for a given **user** and **context**(query, device, etc.)”

<br>

### Metrics
#### Offline metrics
Let’s first go over ROC-AUC, which is a commonly used metric for model comparison in binary classification tasks. However, given that **the system needs well-calibrated prediction scores**, AUC, has the following shortcomings in this ad prediction scenario.
- AUC is insensitive to well-calibrated probabilities.

**Calibration** measures the ratio of average predicted rate and average empirical rate. In other words, it is the ratio of the number of expected actions to the number of actually observed actions.
 
**Why do we need calibration?**
- When we have a significant class imbalance, i.e., the distribution is skewed towards `positive` and `negative` class, we calibrate our model to estimate the likelihood of a data point belonging to a class.
- In our case, we need the model’s predicted score to be **well-calibrated to use in Auction**, 
  
**Log Loss**

Hence, AUC is not a good metric, we need a calibration-sensitive metric. **Log loss** should be able to capture this effectively as Log loss (or more precisely **cross-entropy loss**) is the measure of our predictive error.
- This metric captures to what degree expected probabilities diverge from class labels. As such, it is an absolute measure of quality, which accounts for generating well-calibrated, probabilistic output.
- Let’s consider a scenario that differentiates why log loss gives a better output compared to AUC. 
  - If we multiply all the predicted scores by a factor of `2` and our average prediction rate is double than the empirical rate, **AUC** won’t change but **log loss** will go down.
$$
-\frac{1}{N} \sum_{i=1}^{N}\left[y_{i} \log p_{i}+\left(1-y_{i}\right) \log \left(1-p_{i}\right)\right]
$$
#### Online metrics
**Overall revenue**

This captures the overall revenue generated by the system for the cohorts of the user in either an experiment or, more generally, to measure the overall performance of the system. 

It’s important to call out that just measuring revenue is a very short term approach, as we may not provide enough value to advertisers and they will move away from the system. 

Revenue is basically computed as the sum of the winning bid value (as selected by auction) when the predicted event happens, e.g., if the bid amounts to `$0.5` and the user clicks on the ad, the advertiser will be charged `$0.5`. The business won’t be charged if the user doesn’t click on the ad.

**Overall ads engagement rate**

Engagement rate measures the overall action rate, which is selected by the advertiser.

Some of the actions might be:
- `Click rate`: This will measure the ratio of user clicks to ads.
- `Downstream action rate`: This will measure the action rate of a particular action targeted by the advertiser e.g. add to cart rate, purchase rate, message rate etc.

**Counter metrics**

It’s important to track counter metrics to see if the ads are negatively impacting the platform. There is a risk that users can leave the platform if ads degrade the experience significantly.

So, for online ads experiments, we should track key platform metrics, e.g., for search engines, is `session success` going down significantly because of ads? Are the average queries per user impacted? Are the number of returning users on the platform impacted? These are a few important metrics to track to see if there is a significant negative impact on the platform.

Along with top metrics, it’s important to track direct negative feedback by the user on the ad such as providing following feedback on the ad:
- Hide ad
- Never see this ad
- Report ad as inappropriate

These negative sentiments can lead to the perceived notion of the product as negative.

<br>

### Architectural Components
#### Overview
There will be two main actors involved in our ad prediction system - `platform users` and `advertiser`. Let’s see how they fit in the architecture:

**1. Advertiser flow**

Advertisers create ads containing their content as well as targeting, i.e., scenarios in which they want to trigger their ads. A few examples are:
- `Query-based targeting`: This method shows ads to the user based on the query terms. The query terms can be a partial match, full match, expansion, etc.
- `User-based targeting`: The ads will be subjective to the user based on a specific region, demographic, gender, age, etc.
- `Interest-based targeting`: This method shows interest-based ads. 
  - Assume that on Facebook, the advertiser might want to show ads based on certain interest hierarchies. For example, the advertiser might like to show sports-related ads to people interested in sports.
- `Set-based targeting`: This type shows ads to a set of users selected by the advertisers. 
  - For example, showing an ad to people who were previous buyers or have spent more than ten minutes on the website. 

**2. User flow**

As the platform user queries the system, it will look for all the potential ads that can be shown to this user based on different targeting criteria used by the advertiser. The flow of information will have **two major steps** as described below:
- (1) Advertisers create ads providing targeting information, and the ads are stored in the ads index.
- (2) When a user queries the platform, ads can be selected from the index based on their information (e.g., demographics, interests, etc.) and run through our ads prediction system.

<img src="imgs/Ads_prediction_arch.png" width = "800" height = "310">

<br>

#### Ad selection
The ad selection component will fetch the **top k ads** based on relevance (subject to the `user context`) and `bid` from the ads index.

#### Ad prediction
The ad prediction component will predict user engagement with the ad (the probability that an action will be taken on the ad if it is shown), given the `ad, advertiser, user`, and `context`. Then, it will rank ads based on relevance score and bid.

#### Auction
The auction mechanism then determines:
- whether these top K relevant ads are shown to the user
- the order in which they are shown
- the price the advertisers pay if an action is taken on the ad.

For every ad request, an auction takes place to determine which ads to show. 
- The top relevant **ads selected by the ad prediction system** are given as **input to Auction**. 
- Auction then looks at total value based on an ad’s `bid` as well as its `relevance score`. An ad with the highest total value is the winner of the auction. The total value depends on the following factors:
  - `Bid`: The bid an advertiser places for that ad. In other words, the amount the advertiser is willing to pay for a given action such as click or purchase.
  - `User engagement rate`: An estimate of user engagement with the ad.
  - `Ad quality score`: An assessment of the quality of the ad by taking into account feedback from people viewing or hiding the ad.
  - `Budget`: The advertiser’s budget for an ad

The estimated `user engagement` and `ad quality rates` combined results in the `ad relevance score`. 

The rank order is calculated based on predicted ad score (from the ad prediction component) multipled by the bid. Let’s call this the ad rank score:
$$
\text{Ad rank score} = \text{Ad relevance score} \times \text{bid}
$$
Ads with the highest ad rank score wins the auction and are shown to the user. Once an ad wins the auction, the **cost per engagement (CPE)** or **cost per click (CPC)** will depend on `its ad rank score` and `ad rank score of the ad right below it` in rank order:
$$
CPE = \frac{\text { Ad rank of ad below }}{\text { Ad rank score }}+0.01
$$
- A general principle is that the **ad will cost the minimal price** that still allows it to win the auction.

##### Pacing
- Pacing an ad means evenly spending the ad budget over the selected time period rather than trying to spend all of it at the start of the campaign.
- Remember that whenever the user shows engagement with an ad, the advertiser gets charged the bid amount of the next ad. **If the ad rank score is high, the ad set can spend the whole budget at the start of a time period** (like the start of a new day, as advertisers generally have daily budgets). This would result in a high `cost per click (CPC)` when the ad load (the user engagement rate) is high.
- **Pacing overcomes this by dynamically changing the bid such that the ad set is evenly spread out throughout the day** and the advertiser gets maximum return on investment(ROI) on their campaign. This also prevents the overall system from having a significantly high load at the start of the day, and the budget is spent evenly throughout the campaign.

#### Funnel model approach
- The **ads selection** component selects all the ads that match the targeting criteria (`user demographics` and `query`) and uses a **simple model** to predict the ad’s relevance score.
- The **ads selection** component ranks the ads according to $r$, where $r = bid \times relevance$ and sends the top ads to our ads prediction system.
- The **ads prediction** component will go over the selected ads and uses a highly optimized ML model to predict a **precise calibrated score**.
- The **ads auction** component then runs the auction algorithm based on the bid and predicted ads score to select the top most relevant ads that are shown to the user.

<br>

### 



<br>

###

<br>

###

<br>

###

<br>

## Entity Linking System

<br>

# Recommender System


<br>


<br>
<br>
<br>
<br>
<br>
<br>