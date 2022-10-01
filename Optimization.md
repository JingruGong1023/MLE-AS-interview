## Table of Content

- [Data Augmentation](#Data-Augmentation)
- [Model Architecture](#Model-Architecture)
- [Gradient Descent](#Gradient Descent)
  - [Batch GD](#Batch-GD)
  - [Stochastic GD](#Stochastic-GD)
  - [mini batch GD](#Mini-Batch-GD)
- [Add up to GD: Momentum](#Add_up_to_GD_Momentum)
- [Nestorov’s Accelerated Gradient](#Nestorov’s_Accelerated_Gradient)
- [Adjusting the learning rate](#Adjusting_the_learning_rate)
  - [RMSprop](#RMSprop)

- [Adam: RMSprop with momentum](#Adam_RMSprop_with_momentum)
- [Regularization](#Regularization)
  - [Dropout](#Dropout)
  - [Batch Normalization](#Batch Normalization)
  - [Weight Decay](#Weight_Decay)


</br>



## Setting up a problem

- **Obtain training data** 

  - Use appropriate representation for inputs and outputs 

- **Choose network architecture** 

  - More neurons need more data 
  - Deep is better, but harder to train 

- **Choose the appropriate divergence function** 

  - Choose regularization 

- **Choose heuristics (batch norm, dropout, etc.) **

- **Choose optimization algorithm **

  - – E.g. ADAM • 

- **Perform a grid search for hyper parameters (learning rate, regularization parameter, …) on held-out data** 

- **Train** 

  - Evaluate periodically on validation data, for early stopping if required

    </br>

## Data Augmentation 

It’s obvious how important data is for deep learning. We need clean large datasets for obtaining high accuracies especially for tasks such as machine translation, sentiment analysis, etc. 

The amount of data required also depends on the dimensionality of the data. **The higher the dimensionality, the higher is the amount of data needed.** Data collected can also often be noisy, unannotated, or downright unusable. There are various pre-processing techniques you can use to obtain desired datasets.

You can use image augmentation techniques such as flipping, rotating, Scaling, Changing perspective etc

https://pytorch.org/audio/master/tutorials/audio_feature_augmentation_tutorial.html

<img src="imgs/Screen Shot 2022-09-16 at 10.57.26 PM.png" alt="Screen Shot 2022-09-16 at 10.57.26 PM" height = "60%" width = "60%" />



### GAN 

https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/



## Model Architecture

Experimenting with finding the right model architecture is not easy - and takes a lot of time in a deep learning task.

One of the ways to find useful architectures for your tasks is to to read papers, blogs, Piazza (for this course) and understand the intricacies and measure relative performance for your task. 

Typically, Deeper and wider models tend to generally perform better 

#### Some common architectures

<img src="imgs/Screen Shot 2022-09-16 at 11.01.28 PM.png" alt="Screen Shot 2022-09-16 at 11.01.28 PM" height = "30%" width = "30%" />

<img src="imgs/Screen Shot 2022-09-16 at 11.01.13 PM.png" alt="Screen Shot 2022-09-16 at 11.01.13 PM" height = "30%" width = "30%" />

<img src="imgs/Screen Shot 2022-09-16 at 11.01.35 PM.png" alt="Screen Shot 2022-09-16 at 11.01.35 PM" height = "30%" width = "30%"/>

<img src="imgs/Screen Shot 2022-09-16 at 11.01.22 PM.png" alt="Screen Shot 2022-09-16 at 11.01.22 PM" height = "40%" width = "40%"/>



## Gradient Descent

`Gradient descent` is an **iterative first-order** optimisation algorithm used to find a local minimum/maximum of a given function. This method is commonly used in ML and DL to **minimize a cost/loss function**. 

GD does not work for all functions. There are two specific requirements. A function has to be:

- Differentiable - If a function is differentiable it has a derivative for each point in its domain 
- Convex - we can find a global minimum

**`Gradient`**

Intuitively it is a **slope of a curve at a given point in a specified direction.**

In the case of **a univariate function**, it is simply the **first derivative at a selected point**. In the case of **a multivariate function**, it is a **vector of derivatives** in each main direction (along variable axes). Because we are interested only in a slope along one axis and we don’t care about others these derivatives are called **partial derivatives**.

It is the sum or average of derivatives computed for each training instance

we can see it as the slope of a curve at a given point as well, it measures the changes in parameters w.r.t the change in error

**Since the gradient gives us the direction of steepest ascent, we should go to the opposite direction to find the min**

**`Learning rate`**

The size of each step is called the learning rate. 

With a **high learning rate** we can cover more ground each step, but we risk **overshooting** the lowest point since the slope of the hill is constantly changing. With a very **low learning rate**, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is **time-consuming**, so it will take us a very long time to get to the bottom.

<img src="imgs/Screen Shot 2022-09-16 at 10.50.41 PM.png" alt="Screen Shot 2022-09-16 at 10.50.41 PM" height = "40%" width = "40%" />

### Batch GD:

We take the average of the gradients of all the training examples, and then use that mean gradient to update our parameters. So that's just one step of gradient descent in one epoch. We may need many steps to achieve our minimum, and many epoches to optimize our parameters. Batch Gradient Descent is great for convex or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution

However, it's not working on large dataset.

#### Decaying learning rate

**Learning rate must shrink with time for convergence**

<img src="imgs/Screen Shot 2022-09-19 at 10.38.52 PM.png" alt="Screen Shot 2022-09-19 at 10.38.52 PM" height = "50%" width = "50%" />

A common approach (for nnets): 

1. Train with a fixed learning rate until loss (or performance on a held-out data set) stagnates 

2. decaying learning rate

   <img src="imgs/Screen Shot 2022-09-19 at 10.41.01 PM.png" alt="Screen Shot 2022-09-19 at 10.41.01 PM" height = "50%" width = "50%" />

3. Return to step 1 and continue training from where we left off



### Stochastic GD

In Stochastic Gradient Descent (SGD), we consider just one example at a time to take a single step. 

We do the following steps in **one epoch** for SGD:

1. Take an example
2. (Feed it to Neural Network)
3. Calculate it’s gradient
4. Use the gradient we calculated in step 3 to update the weights
5. Repeat steps 1–4 for all the examples in training dataset

Since we are considering just one example at a time the cost will fluctuate over the training examples and it will **not** necessarily decrease. But in the long run, you will see the cost decreasing with fluctuations.

`Problem with SGD`

Correcting the functions for individual instances will lead to never-ending updates

Also because the cost is so fluctuating, it **will never reach the minima** but it will keep dancing around it.

**Potential solution** (cannot totally solve it, just make it better): we can shrink learning rate with iterations to prevent this

corrections for individual instances with the eventual miniscule learning rate will not modify the function

**The fastest converging series for the learning rate is**: learning rate at k th epoch = 1/k

- **If the loss is convex**, SGD converges to the optimal solution
- **If the loss is not convex**, SGD converges to the local minimum
- SGD converges "almost surely" to a global or local minimum for most functions

SGD can be used for **larger datasets**. It converges faster when the dataset is large as it causes updates to the parameters more frequently.



### Mini Batch GD

`Batch Gradient Descent` can be used for **smoother curves**.`SGD` can be used when the dataset is large. Batch Gradient Descent converges directly to minima. SGD converges faster for larger datasets. But, since in SGD we use only one example at a time, we cannot implement the vectorized implementation on it. This can slow down the computations. To tackle this problem, a mixture of Batch Gradient Descent and SGD is used.

We use a batch of a **fixed number** of training examples which is less than the actual dataset and call it a **mini-batch**. Doing this helps us achieve the advantages of both the former variants we saw. 

So, after creating the mini-batches of fixed size, we do the following steps in **one epoch:**

1. Pick a mini-batch
2. Feed it to Neural Network
3. Calculate the mean gradient of the mini-batch
4. Use the mean gradient we calculated in step 3 to update the weights
5. Repeat steps 1–4 for the mini-batches we created

Just like SGD, the average cost over the epochs in mini-batch gradient descent fluctuates because we are averaging a small number of examples at a time. 

<img src="imgs/Screen Shot 2022-09-16 at 10.48.46 PM.png" alt="Screen Shot 2022-09-16 at 10.48.46 PM" height = "30%" width = "30%" />

So, when we are using the mini-batch gradient descent we are updating our parameters frequently as well as we can use vectorized implementation for faster computations.

Generally, **mini batch can give the as low loss as batch , but as fast as SGD**

- The number of mini-batch is generally set to the **largest** that your hardware will support (in memory) without compromising overall compute time
  - Larger minibatches = less variance  -> better minima
  - Larger minibatches = few updates per epoch -> faster
- Convergence depends on learning rate
  - Simple technique: fix learning rate until the error plateaus, then reduce learning rate by a fixed factor (e.g. 10)

- higher batch sizes leads to lower asymptotic test accuracy
- we can recover the lost test accuracy from a larger batch size by increasing the learning rate

#### Compare these methods:

**Speed**: SGD > mini batch > batch

**Acuracy**: batch > mini batch >SGD



### Add up to GD: Momentum

<img src="imgs/Screen Shot 2022-09-24 at 11.04.58 PM.png" alt="Screen Shot 2022-09-24 at 11.04.58 PM" height = "50%" width = "50%" />

The problem with GD is that it's a first order optimization method, which means, although it can tell us the direction and whether the loss increased or not, it cannot tell us the curvature of the loss function. In other words, the weight update at a moment t is governed by the learning rate and gradient at that moment only. It does not take into account the past steps.

**Incremental SGD and mini-batch gradients tend to have high variance**

By adding a momentum term in the gradient descent, it updates parameters with weighted moving average, gradients accumulated from past iterations will push the cost further to move around saddle point  

**The momentum method maintains a running average of all gradients until the current step**, with the intuition

- In directions in which the convergence is smooth, the average will have a large value 

- In directions in which the estimate swings, the positive and negative swings will cancel out in the average

  <img src="imgs/Screen Shot 2022-09-24 at 11.19.33 PM.png" alt="Screen Shot 2022-09-24 at 11.19.33 PM" height = "50%" width = "50%" />

Increase $\beta$ will make the line smoother. but if $\beta$ is too big, it could also smooth out the update too much

**The steps in Momentum**:

1. First computes the gradient step at the current location (blue)

2. Then adds in the scaled previous step , Which is actually a running average(green)

3. To get the final step(red)

   <img src="imgs/Screen Shot 2022-09-24 at 11.23.58 PM.png" alt="Screen Shot 2022-09-24 at 11.23.58 PM" height = "50%" width = "50%" />



### Nestorov’s Accelerated Gradient

***An update from Momentum***

Change the order of operations

At any iteration, to compute the current step: 

	1. First extend the previous step
	1. Then compute the gradient step at the resultant position
	1. Add the two to obtain the final step

<img src="imgs/Screen Shot 2022-09-24 at 11.26.50 PM.png" alt="Screen Shot 2022-09-24 at 11.26.50 PM" height = "50%" width = "50%" />



## Adjusting the learning rate

<img src="imgs/Screen Shot 2022-09-24 at 11.52.15 PM.png" alt="Screen Shot 2022-09-24 at 11.52.15 PM"  height = "70%" width = "70%"/>**Have separate learning rates for each component** 

- Directions in which the derivatives swing more should likely have lower learning rates 
  - Is likely indicative of more wildly swinging behavior 
- Directions of greater swing are indicated by total movement 
  - Direction of greater movement should have lower learning rate



### RMSprop

RMSprop uses an **adaptive learning rate** instead of treating the learning rate as a hyperparameter. This means the learning rate changes over time.

Using a moving average of squared gradients to normalize the gradient. This normalization balances the step size

**Decreasing the step for large gradient to avoid exploding and increasing the step for small gradients to avoid vanishing**

<img src="imgs/Screen Shot 2022-09-24 at 11.59.39 PM.png" alt="Screen Shot 2022-09-24 at 11.59.39 PM" height = "30%" width = "30%" />



## Adam: RMSprop with momentum

- `RMS` prop only adapts the learning rate 
- `Momentum` only smooths the gradient 
- `ADAM` combines the two

**Procedure**

1. Maintain a running estimate of the mean derivative for each parameter
2. Maintain a running estimate of the mean squared value of derivatives for each parameter 
3. Scale update of the parameter by the inverse of the root mean squared derivative

<img src="imgs/Screen Shot 2022-09-25 at 12.03.44 AM.png" alt="Screen Shot 2022-09-25 at 12.03.44 AM" height = "30%" width = "30%"  />

typically, $\gamma$ = 0.999, and $\delta$ = 0.9 so the two terms doesn't dominate early learning



</br>

## Regularization

Let’s think briefly about what we expect from a good predictive model. We want it to peform well on unseen data. Classical generalization theory suggests that to close the gap between train and test performance, we should aim for a simple model. Simplicity can come in the form of a small number of dimensions.

### Dropout

The term “dropout” refers to dropping out the neurons **(input and hidden layer)** in a neural network. **All the forward and backwards connections** with a dropped node are temporarily removed, thus creating a new network architecture out of the parent network. 

**Different nodes are dropped by a dropout probability of p during each forward pass.** This leads to complex co-adaptations, which in turn reduces the overfitting problem because of which the model generalises better on the unseen dataset

<img src="imgs/Screen Shot 2022-09-16 at 11.09.01 PM.png" alt="Screen Shot 2022-09-16 at 11.09.01 PM" height = "30%" width = "30%" />

It means we dropped some nodes in each layer randomly with specified probabilities, the dropped nodes are randomly chosen for each iteration to make our NN model simpler.
For layer with a lot nodes, we can set the keep-prob low, because there is a high chance of overfitting. For layer with just a little nodes, we can set the keep-prob high, for example, for the output layer with only one node, we don’t need to worry about the overfitting, so that we can set the keep-prob = 1

With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.

**`Dropout` is a stochastic data/model erasure method that sometimes forces the network to learn more robust models**

#### Dropout during test: 

Instead of multiplying every output by $\alpha$ , multiply all weights by $\alpha$

$ W_{test}\ = \ \alpha*W_{trained}$



### Early Stopping

<img src="imgs/Screen Shot 2022-09-25 at 11.43.23 AM.png" alt="Screen Shot 2022-09-25 at 11.43.23 AM" height = "50%" width = "50%" />

Continued training can result in over fitting to training data

- Track performance on a held-out validation set
-  Apply one of several early-stopping criterion to terminate training when performance on validation set degrades significantly



### Gradient Clipping

<img src="imgs/Screen Shot 2022-09-25 at 11.53.42 AM.png" alt="Screen Shot 2022-09-25 at 11.53.42 AM" height = "50%" width = "50%" />

When the divergence has a steep slope, the derivative can be too high

This might leed to instability

`Gradient Clipping` set a ceiling on derivative value

$if \ d_wD \ > \ \theta \ then \ d_wD \ = \theta$

typically $\theta$ = 5



### Batch Normalization

Wildly successful and simple technique for **accelerating training and learning better neural network representations**

- The key motivation behind BatchNorm is internal covariate shift. It is defined as the change in the distribution of network activations due to the change in network parameters during training. -> **Since we assume all the training data are similarly distributed. but the fact might not like this** -> we might have different distribution for each mini batch

- At every epoch of training, **weights are updated** and a different minibatch is being processed, which means that the inputs to a neuron is slightly different every time.

- As these changes get passed on to the next neuron, it creates a situation **where the input distribution of every neuron is different at every epoch**. This co-adaptation problem, significantly slows learning. 

- Hence, these shifts in input distributions can be problematic for neural networks, especially deep neural networks that could have a large number of layers. Which means, if each layer is normalised, **the weight changes made by the previous layer and noise between data is partially lost**, as some non-linear relationships are lost during normalisation. This can lead to **suboptimal weights** being passed on.

- To fix this, batch normalisation adds two learnable parameters, gamma γ and beta β, which can scale and shift the normalised value.

  <img src="imgs/Screen Shot 2022-09-16 at 11.43.11 PM.png" alt="Screen Shot 2022-09-16 at 11.43.11 PM" height = "40%" width = "40%" />

**There are two main steps in Batch Normalization**:

1. Move all batches to have a mean of 0 and unit sd (normalize)

2. Then move the entire collection to the appropriate location

   <img src="imgs/Screen Shot 2022-09-21 at 4.44.25 PM.png" alt="Screen Shot 2022-09-21 at 4.44.25 PM" height = "30%" width = "30%" />

Batch normalization is a shift-adjustment unit that happens **after the weighted addition of inputs but before the application of activation**

 - Is done independently for each unit, to simplify computation

   <img src="imgs/Screen Shot 2022-09-21 at 4.46.28 PM.png" alt="Screen Shot 2022-09-21 at 4.46.28 PM" height = "70%" width = "70%"/>

- $\beta \ and \ \gamma$ are neuron specific

#### Backpropagation in batch normalization

<img src="imgs/Screen Shot 2022-09-21 at 4.50.56 PM.png" alt="Screen Shot 2022-09-21 at 4.50.56 PM" height = "70%" width = "70%"/>

If you don't have enough diversity in your mini batch, ie, if all instances are similar in one batch, derivative of Loss w.r.t zi will go to zero -> **we don't want to use batch normalization in SGD**

-  **Batch normalization may only be applied to some layers** 
  - Or even only selected neurons in the layers
- **Improves both convergence rate and neural network performance** 
- **To get maximum benefit from BN**, learning rates must be increased and learning rate decay can be faster 
  - Since the data generally remain in the high-gradient regions of the activations 
  - Also needs better randomization of training data order

During training, the mean and standard deviation are calculated using samples in the mini-batch. However, in testing, it does not make sense to calculate new values. Hence, batch normalisation uses a **running mean and running variance in testing** that is calculated during training

#### Benefits

1. Higher training rate
2. Faster Convergence
3. Regularization



### weight decay

***Also called L2 Regularization***



​												$\min _{w \in \mathbb{R}^{p}, b \in \mathbb{R}} \sum_{i=1}^{n}\left(w^{T} \mathbf{x}_{i}+b-y_{i}\right)^{2}+\alpha\|w\|^{2}$

- Always has a **unique** solution.
- Tuning parameter $\alpha$.

**What is L2-regularization actually doing?**:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

In Ridge regression we add another term to the optimization problem

- Not only do we want to fit the training data well, we also want $w$ to have a small squared `l2` norm. 
- The idea here is that we're decreasing the "slope" along each of the feature **by pushing the coefficients towards zero**. -> minimize $w$
- This prevent the model to be too complex(prone to overfitting)

So there are two terms in the objective function of the model: 

- **the data fitting term** here that wants to be close to the training data according to the squared norm, 

- **the penalty or regularization term** here that wants $w$ to have small norm, and that doesn't depend on the data.

- Usually these two goals are somewhat **opposing**

  - If we made $w$ zero, the second term would be zero, but the predictions would be bad. So we need to **trade off** between these two. 
  - If we set $\alpha$ to zero, we get normal linear regression

  This is a very typical example of a general principle in machine learning, called **regularized empirical risk minimization**

  - Many models like `linear models, SVMs, neural networks` follow the general framework of empirical risk minimization
  - We formulate the machine learning problem as an optimization problem over a family of functions. In our case that was **the family of linear functions parametrized** by $w$ and $b$

  **LR Coefficients**

  - Notice: **depending on how much you regularize** the direction of effect goes in opposite directions, what that tells me is **don't interpret your models too much** because clearly, either it has a **positive** or **negative** effect, it can't have both.

<img src="../MLE:AS interview/imgs/Screen Shot 2022-09-27 at 7.04.45 PM.png" alt="Screen Shot 2022-09-27 at 7.04.45 PM" height = "60%" width = "60%" />