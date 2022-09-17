## Table of Content

- [Gradient Descent](#Gradient Descent)

  - [Batch GD](#Batch-GD)

  - [Stochastic GD](#Stochastic-GD)

  - [mini batch GD](#Mini Batch GD)

    



## Gradient Descent

`Gradient descent` is an **iterative first-order** optimisation algorithm used to find a local minimum/maximum of a given function. This method is commonly used in ML and DL to **minimize a cost/loss function**. 

GD does not work for all functions. There are two specific requirements. A function has to be:

- Differentiable - If a function is differentiable it has a derivative for each point in its domain 
- Convex - we can find a global minimum

**`Gradient`**

Intuitively it is a **slope of a curve at a given point in a specified direction.**

In the case of **a univariate function**, it is simply the **first derivative at a selected point**. In the case of **a multivariate function**, it is a **vector of derivatives** in each main direction (along variable axes). Because we are interested only in a slope along one axis and we don’t care about others these derivatives are called **partial derivatives**.

**`Learning rate`**

The size of each step is called the learning rate. 

With a **high learning rate** we can cover more ground each step, but we risk **overshooting** the lowest point since the slope of the hill is constantly changing. With a very **low learning rate**, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is **time-consuming**, so it will take us a very long time to get to the bottom.

## Gradient Descent

`Gradient descent` is an **iterative first-order** optimisation algorithm used to find a local minimum/maximum of a given function. This method is commonly used in ML and DL to **minimize a cost/loss function**. 

GD does not work for all functions. There are two specific requirements. A function has to be:

- Differentiable - If a function is differentiable it has a derivative for each point in its domain 
- Convex - we can find a global minimum

**`Gradient`**

Intuitively it is a **slope of a curve at a given point in a specified direction.**

In the case of **a univariate function**, it is simply the **first derivative at a selected point**. In the case of **a multivariate function**, it is a **vector of derivatives** in each main direction (along variable axes). Because we are interested only in a slope along one axis and we don’t care about others these derivatives are called **partial derivatives**.

**`Learning rate`**

The size of each step is called the learning rate. 

With a **high learning rate** we can cover more ground each step, but we risk **overshooting** the lowest point since the slope of the hill is constantly changing. With a very **low learning rate**, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is **time-consuming**, so it will take us a very long time to get to the bottom.

<img src="../Machine_Learning/imgs/Screen Shot 2022-08-12 at 1.12.24 PM.png" alt="Screen Shot 2022-08-12 at 1.12.24 PM" height = "50%" width = "50%" />

### Batch GD:

We take the average of the gradients of all the training examples, and then use that mean gradient to update our parameters. So that's just one step of gradient descent in one epoch. We may need many steps to achieve our minimum, and many epoches to optimize our parameters. Batch Gradient Descent is great for convex or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution

However, it's not working on large dataset.

### Stochastic GD

In Stochastic Gradient Descent (SGD), we consider just one example at a time to take a single step. 

We do the following steps in **one epoch** for SGD:

1. Take an example
2. (Feed it to Neural Network)
3. Calculate it’s gradient
4. Use the gradient we calculated in step 3 to update the weights
5. Repeat steps 1–4 for all the examples in training dataset

Since we are considering just one example at a time the cost will fluctuate over the training examples and it will **not** necessarily decrease. But in the long run, you will see the cost decreasing with fluctuations.

<img src="../Machine_Learning/imgs/Screen Shot 2022-08-12 at 1.18.06 PM.png" alt="Screen Shot 2022-08-12 at 1.18.06 PM" height = "30%" width = "30%"/>

Also because the cost is so fluctuating, it **will never reach the minima** but it will keep dancing around it.

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

<img src="imgs/Screen Shot 2022-09-15 at 11.07.22 PM.png" alt="Screen Shot 2022-09-15 at 11.07.22 PM" height = "50%" width = "50%"/>

So, when we are using the mini-batch gradient descent we are updating our parameters frequently as well as we can use vectorized implementation for faster computations.



### Batch GD:

We take the average of the gradients of all the training examples, and then use that mean gradient to update our parameters. So that's just one step of gradient descent in one epoch. We may need many steps to achieve our minimum, and many epoches to optimize our parameters. Batch Gradient Descent is great for convex or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution

However, it's not working on large dataset.

### Stochastic GD

In Stochastic Gradient Descent (SGD), we consider just one example at a time to take a single step. 

We do the following steps in **one epoch** for SGD:

1. Take an example
2. (Feed it to Neural Network)
3. Calculate it’s gradient
4. Use the gradient we calculated in step 3 to update the weights
5. Repeat steps 1–4 for all the examples in training dataset

Since we are considering just one example at a time the cost will fluctuate over the training examples and it will **not** necessarily decrease. But in the long run, you will see the cost decreasing with fluctuations.

<img src="../Deep_Learning/NLP/image/Screen Shot 2022-08-12 at 1.18.06 PM.png" alt="Screen Shot 2022-08-12 at 1.18.06 PM" height = "50%" width = "30%" />

Also because the cost is so fluctuating, it **will never reach the minima** but it will keep dancing around it.

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

<img src="../Deep_Learning/NLP/image/Screen Shot 2022-08-12 at 1.22.14 PM.png" alt="Screen Shot 2022-08-12 at 1.22.14 PM" height = "30%" width = "30%" />

So, when we are using the mini-batch gradient descent we are updating our parameters frequently as well as we can use vectorized implementation for faster computations.