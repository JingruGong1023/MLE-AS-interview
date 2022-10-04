### Basic ML

1. **What is overfitting?**

   when model is too complex and we got training performance much better than testing

2. **What is underfitting**

   when model is too simple and we got bad training performance 

3. **what is the trade-off between bias/variance**

   the bias–variance tradeoff is the property of a model that the variance of the parameter estimated across samples can be reduced by increasing the bias in the estimated parameters

4. **How to deal with overfitting**

   based on different models. general approach would be regularization, such as dropout, l1, l2, and early stopping, to make the model simpler

   or adding more data

5. **difference between Generative and Discrimitive**

   Discriminative models draw boundaries in the data space, while generative models try to model how data is placed throughout the space

   A generative model focuses on explaining how the data was generated, while a discriminative model focuses on predicting the labels of the data.

   <img src="imgs/Screen Shot 2022-09-24 at 10.09.38 PM.png" alt="Screen Shot 2022-09-24 at 10.09.38 PM" height = "50%" width = "50%" />

   examples of discriminative: SVM, RF, logistic

   examples of generative: GANs, Bayesian

6. **what is relative entropy / crossentropy, and K-L divergence, their intuition**

   Cross entropy(logloss) is used to measure the uncertainty of a system

   loss = average divergence

   KL is popular for networks that perform classification, the output is a probability vector

   <img src="imgs/Screen Shot 2022-09-24 at 10.20.56 PM.png" alt="Screen Shot 2022-09-24 at 10.20.56 PM" height = "30%" width = "30%" />

   The cross entropy is equivalent to KL in binary problem

7. 

8. 

9. 



### Deep Learning 

1. **Why DNN has bias term, what is the intuition of it**

   The bias in NN is used to shift our activation function, this can be useful when the desired output distribtion is not centered at zero, which is the common case

2. **effect of learning rate**

   learning rate is applied while we optimize our model, generally, learning rate determines how fast our model learns

   if learning rate too large, we might overshoot

   if learning rate too small, we might learn too slow, and even worse, might stuck in local min

   we generally start with large learning rate and use scheduler to decrease it while training

3. **What is back propagation**

   we use back propagation to calculate the gradients to use in gradient descent

4. **Can we initialize weights to 0? why and why not**

   no, because that will cause the hidden layer symmetric

5. **common activation functions （sigmoid, tanh, relu, leaky relu)**

   **Sigmoid**: only when you need output to be binary, might lead to gradient vanish, Converge very slowly when input z is really large(gradient small)

   **Tanh**: better than sigmoid, derivative more steep. but it might work worse, because the mean of its output is close to 0, making learning more complex for next layer. Converge very slowly when input z is really large

   **ReLU**: speed up training, usually used in hidden layers, but you have to let the gradients vanish

   **Leaky ReLU**: can be the first solution to the problem of the gradients’ vanish.

   **Softmax**: Usually used in output layer, when we have multiple classes the predict

6. **What is gradient vanishing**

   Some activations, such as sigmoid function, squishes a large input into the range[0,1], therefore, a large change in input will result small change in the output, which will cause the gradient to be small, and with more layers added, the gradient will be even smaller. in the backpropagation, This issue will cause the later gradient unable to affect gradients in the early layers, which will make the update ineffective

   Solution: , use another activation function, such as relu, LSTM

7. **gradient exploding**

   In deep networks or recurrent neural networks, error gradients can accumulate during an update and result in very large gradients. These can cause the update unstable, even cause the weights to overflow

   Solution: gradient clipping, early stopping

8. **What is Gradient Descent**

   GD is an iterative first-order optimization algorithm used to find the local min /max of a given function.

9. **What kind of function can GD work on**

   differentiable, convex

10. **What is Gradient**

    It is the sum or average of derivatives computed for each training instance

    we can see it as the slope of a curve at a given point as well, it measures the changes in parameters w.r.t the change in error

    Since the gradient gives us the direction of steepest ascent, we should go to the opposite direction to find the min

11. **What is Learning rate**

    The size of each step in GD

12. **Trade-off between high and low Learning rate**

    high learning rate: learn faster, might help us jump out of a saddle point, but might overshoot

    low learning rate: learn slowly, but we can confidently move in the direction

13. **Batch GD**

    for each epoch , for all instances, Calculate average gradient in backpropagation, and use the mean gradient to update parameters. 

14. **What is decaying learning rate**

    in actual practice, with the trade-off between high and low learning rate, we usually begin with a fairly large learning rate, and reduce it while training

15. **What is SGD**

    instead of taking average of all training instances, for SGD, we only take one instance and update weights one time for one instance. 

    The cost will fluctuate a lot and might never end updating, and might never reach the global min

    but it converges faster than Batch GD

16. **What is Mini batch GD**

    We use a batch of a **fixed number** of training examples which is less than the actual dataset and call it a **mini-batch**. Doing this helps us achieve the advantages of both the former variants we saw. 

17. **batch size in minibatch GD**

    usually we set as large batch size as our computer can handle, lager batch size usually leads to less variance and better minima, and runs faster

18. **What is the problem with GD as a first order optimization**

    The problem with GD is that it's a first order optimization method, which means, although it can tell us the direction and whether the loss increased or not, it cannot tell us the curvature of the loss function. In other words, the weight update at a moment t is governed by the learning rate and gradient at that moment only. It does not take into account the past steps.

19. **What is momentum doing**

    By adding a momentum term in the gradient descent, it updates parameters with weighted moving average, gradients accumulated from past iterations will push the cost further to move around saddle point  , make the gradients with less variance

20. **What is the intuition behind momentum**

    In directions in which the convergence is smooth, the moving average will have a large value 

21. **What is the hyperparameter in momentum**

    $\beta$ : increase it will make the updates smoother, but might smooth out too much

22. **The steps in momentum**

    1. First computes the gradient step at the current location 

    2. Then adds in the scaled previous step , Which is actually a running average

    3. To get the final step, sum them up

23. **What is Nestorov's Accelerated Gradient**

    similar as momentum, just change the order of the steps

    1. First extend the previous step
    1. Then compute the gradient step at the resultant position
    1. Add the two to obtain the final step

24. **What is the intuition behind RMSprop**

    RMSprop uses an **adaptive learning rate** instead of treating the learning rate as a hyperparameter. This means the learning rate changes over time.

    Using a moving average of squared gradients to normalize the gradient. This normalization balances the step size

    Decreasing the step for large gradient to avoid exploding and increasing the step for small gradients to avoid vanishing

25. **What is Adam**

    combining rmsprop and momentum, both adapts learning rate and smooths gradients.

    1. Maintain a running estimate of the mean derivative for each parameter
    2. Maintain a running estimate of the mean squared value of derivatives for each parameter 
    3. Scale update of the parameter by the inverse of the root mean squared derivative

26. **How to set Adam hyper params**

    Set them large, typically $\gamma = 0.999, and \ \delta = 0.9$ so that two terms does not dominate early learning

27. **Why do we need regularization**

    We expect a good model to be able to predict unseen data, ie test data, to close the gap between test accuracy and training accuracy, we generally aim for a simpler model, which can come in with small number of dimensions

28. **What is dropout**

    dropping out the neurons (input and hidden layer) in a neural network with a hyperparameter, p , as the dropout probability. **All the forward and backwards connections** with a dropped node are temporarily removed, thus creating a new network architecture out of the parent network. 

29. **Dropout during test**

    We multiply all training weights by dropout probability to get the testing weights

30. **Why dropout works**

    With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.

31. **What is Gradient Clipping**

    when the divergence has a steep slope, the gradient can be too high, leads to instability

    set a ceiling on derivative is a potential way to solve it

32. **What is the intuition behind batch norm**

    The key motivation is internal covariate shift, which is the change of distribution of network activation due to the change in parameters during training. 

    We assume all training data are similarly distributed, but in fact they are not

    The covariate shift can significantly slow the training, might even lead to suboptimal weights for deep nn

33. **How does batch norm work**

    1. Move all batches to have a mean of 0 and unit sd
    2. move the entire collection to the appropriate location

34. **When does BN happen**

    after the calculation of  Z and before activation function

35. **When can't we use BN?**

    SGD, no variance in each batch, because in such case the derivative of loss w.r.t zi will go to zero

36. **What can BN help with**

    1. improve training speed
    2. improve performance

37. **How to get maximum benefit from BN**

    learning rates must be increased and learning rate decay can be faster . Also needs better randomization of training data order

38. **BN in testing**

    batch normalisation uses a **running mean and running variance in testing** that is calculated during training

39. **When transfer learning makes sense?**

    To save time and resources from having to train multiple machine learning models from scratch to complete similar tasks

40. 

41. 

42. 

43. 

44. 

45. 

</BR>

### Natural Language Processing

1. **What is NLP**

   NLP helps transform human languages into machine-read code

2. **What are the general processing techniques**

   tokenization, Lemmatization, stemming, stop words

3. **What is Tokenization**

   split sentences into single words

4. **Lemmatization**

   reduces words into its base form based on dictionary definition , such as are -> be

5. **Stemming**

   reduce words into its base form without context , such as ended -> end

6. **Stop words**

   irrelevant and common words, such as is, and ,the...

7. **Markov Chain**

   stochastic and memoryless process that predicts future events based only on the current state

8. **n-grams**

   predict the next term based on a sequence of n terms based on Markov chain

9. **bag of words**

   represents text using word frequencies, without context/order

10. **tf-idf**

    measures word importance for a document by multiplying the occurances of a term in a document with the inverse document frequency

11. **Cosine Similarity**

    measures similarity between vectors, with output range [0,1]

    $cos(\theta)\ = \ \frac{A*B}{||A||||B||}$

12. **Word Embeddings**

    Maps words and phrases to numerical vectors

13. **Word2vec**

    a framework learning word vectors

    1. given a large corpus
    2. go through each position t in the text, which has its center c and context o
    3. use the similarity calculation of c and o to predict the probability of o given c
    4. keep adjusting the word vectors until reach the max probability

14. **What is a context of a word**

    when a word appears in a document, its context is the set of words nearby

15. **Skip-gram**

    an unsupervised learning to predicts the context given a word

16. **Continuous bag-of-words(CBOW)**:

    predict the word given its context

17. **GloVe**

    learn word similarity by minimizing the difference between (how related the words are) and (How frequency the words appear together)

18. **BERT** https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

    accounts for word order and trains on subwords, and unlike word2vec and GloVe, BERT outputs different vectors for different uses of words, such as cell phone vs blood cell

19. **Sentiment Analysis**

    Extracts the attitudes and emotions from a text

20. **Polarity**

    measures positive, negative or neutral opinions

    such as really fun vs hardly fun

21. **Sentiment**

    measures emotional state such as happy or sad

22. **Subject-object Identification**

    classifies sentences as either subjective or objective

23. **Topic Modeling - LDA**

    Generates k topics by first assigning each word to a random topic, then iteratively updating assignments based on parameters $\alpha$ the mix topic per document, and $\beta$ the distribution of words per topic

24. **Topic Modeling - LSA**

    Identifies patterns using tf-idf scores and reduces data to k dimensions through SVD

25. **What is Recurrent Neural Network**

    predicts sequential data using a temporally connected system that takes both new and previous inputs using hidden states

    RNN can work with different dimensions of input-output combination, many-to-one, many-to-many, etc

    Rely on weight sharing

26. **What is the problem with RNN**

    To reduce redundancy calculation during backpropagation, later gradients are found by chaining previous gradients, however repeatedly multiply values greater or smaller than 1 will lead to exploding gradients or vanishing gradients

27. **What is Long short term Memory**

    Learns long-term dependencies using gated cells and maintains a separate cell state from what is outputed

28. **What are the gates in LSTM**

    Forget Gate: forget irrelevant info from previous layers, or keep from previous layer

    input gate: Controls what parts of the new cell content are written to this cell

    output gate: controls what parts of cell are output to hidden state

29. **What are the states in LSTM**

    New cell content: this is the new content to be written to the cell

    cell state: erase some content from last cell state and write some new cell content

    Hidden state: read/output some content from the cell

30. **What is the difference between GRU and LSTM**

    The Gated Recurrent Unit (GRU) is a type of Recurrent Neural Network (RNN) that, in certain cases, has advantages over long short term memory (LSTM). **GRU uses less memory and is faster than LSTM**,

     however, LSTM is **more accurate** when using datasets with longer sequences.

    GRU does not possess any internal memory, they don’t have an output gate that is present in LSTM

31. **What is Attention model**

    unlike traditional sequence model, usually decreases the accuracy with longer sentences input, attention model is the sequence model that we take into account of some nearby words in the sentence

32. **What kind of RNN used in attention model**

    Bidirectional

33. **What's wrong with traditional sequence model comparing to attention model**

    the normal sequence model only use a fixed-length context vector to explain a word, but it often forgot the earlier part of the sequence once it has processed the entire sequence

34. **How does attention model work**

    We calculate the alpha , the amount of attention we should pay to , for each word in the sentence

    And we do this for every word in the sentence

35. **How do we calculate alpha**

    use a softmax(because we want to use it as weights, so they should sum to 1), with $e^{t,t'}$, which can be learned by a small rnn with previous stage of input and activation as inputs

36. 

37. 

38. 

    

### Convolutional Neural Network

1. **What is Convolutional Layer**

   iterate over windows of an image to apply weights ,bias and activation functions to create feature maps, different window will have different feature maps

2. **What is pooling**

   Downsampling convolutional layer to reduce dimensionality and maintain spatial invariance, allowing detection of features even if they have shifted slightly

   Common technique including returning max or average of a pooling window

3. **General CNN architecture**

   1. perform a series of convolution, ReLU and followed by a pooling layer
   2. Feed output into a fully connected layer to get output















