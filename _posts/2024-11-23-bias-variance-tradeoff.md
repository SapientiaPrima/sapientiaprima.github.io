---
layout: post
title: "Bias-Variance Trade-off"
author: "Discipulus"
categories: journal
tags: [ machine learning, bias, variance, tradeoff, interview]
image: bias-variance-tradeoff.jpg
---

# So, what is Bias Variance Trade-off?

In this post, I'm going to explain bias and variance completely. 

## Conceptual Introduction

At its core, **bias** refers to the error introduced when a model makes overly simplistic assumptions about a complex real-world problem. These assumptions simplify the problem to make it solvable, but they can lead to inaccuracies. A model with high bias often fails to capture the underlying patterns in the data, resulting in **underfitting**. This means the model performs poorly on both the training data and unseen test data.

For example, consider a classification task to identify dog breeds based on certain features like height, weight, and color. If the model assumes all dogs belong to the same breed (say, breed A), it simplifies the problem too much, ignoring the distinctive features that separate breeds. This oversimplification introduces bias.

On the other hand, **variance** measures the sensitivity of a model to fluctuations in the training data. A model with high variance captures noise or irrelevant patterns from the training data, leading to **overfitting**. Such a model may perform exceptionally well on the training data but generalize poorly to new, unseen data.

Returning to the dog classification example, a high-variance model might pick up on irrelevant details, like the background of the training images or the lighting conditions, and mistakenly treat these as important features for classifying breeds. This results in a model that is overly complex and fails to generalize.

## The Math Behind Bias and Variance Trade-off

Let us formalize these concepts mathematically. You have a hypothesis set $H$, which is basically the set of all functions you can accept as your final model. Note that your basic assumption is that there is a function $f$ (the true function) which represents the relation between features and labels. For a classification problem, it means that $f$ can take features of the training samples and map them to the labels of the corresponding samples. In other words, $f$ is the best function you're looking for. However, unfortunately, you don't know $f$. Instead, you have the outputs given by it for some specific inputs, and you call inputs and outputs together "the training set". Denote this set by $D$.

From the functions available in $H$, you will finally choose a model (a function) $h_D$ based on the $D$ you were given and a learning algorithm $A$ which is not our focus in this post. Obviously, since $f$ might not really exist or it may not be a true "function" (e.g., the real behavior of the data might be stochastic, leading to an $f$ which gives different outputs given the same input), you might not find a perfect $h_D$ which is equal to $f$. So, what should you do in this situation?

The answer is simple. You just try your best. But what does "trying your best" mean? It means you should minimize the error of your model. The error of a model is the difference between the output of the model and the true output. To be more precise, you'll have to minimize the expectation of the error, which we denote by $\mathbb{E}_D$ since it depends on the training set $D$. Assuming a fixed input $x$, the expected error would be:

$$ \mathbb{E}_D[(h_D(x) - f(x))^2] $$

Now let us use some tricks to decompose this error. First, we can write it as:

$$ \mathbb{E}_D[(h_D(x) - f(x))^2] = \mathbb{E}_D[(h_D(x) - C + C - f(x))^2] $$

Note that the only difference between the two sides of the equation is the addition of $C$ and the subtraction of it. $C$ is now just a constant and we will define it later. Now, define:

$$A := h_D(x)-C$$

$$B := C-f(x)$$

Then, we can rewrite the error as:

$$ \mathbb{E}_D[A^2] + \mathbb{E}_D[B^2] + 2\mathbb{E}_D[AB] $$

We decomposed the sum since the expectation of the sum is equal to the sum of the expectations. Now, let's focus on the last term. We can write it as:

$$ \mathbb{E}_D[AB] = \mathbb{E}_D[(h_D(x)-C)(C-f(x))] $$

But since $C-f(x)$ is independent of $D$, we can write it as:

$$ \mathbb{E}_D[AB] = (C-f(x))\mathbb{E}_D[h_D(x)-C] $$

Now, let's define $C$ as the expectation of $h_D(x)$:

$$ C := \mathbb{E}_D[h_D(x)] $$

Then, the last term becomes:

$$ \mathbb{E}_D[AB] = (C-f(x))(C-C) = 0 $$

So, the error becomes:

$$ \mathbb{E}_D[A^2] + \mathbb{E}_D[B^2] $$

Let's focus on the first term. We can write it as:

$$ \mathbb{E}_D[A^2] = \mathbb{E}_D[(h_D(x)-C)^2] $$

But since $C$ is the expectation of $h_D(x)$, we can write it as:

$$ \mathbb{E}_D[A^2] = \mathbb{E}_D[(h_D(x)-\mathbb{E}_D[h_D(x)])^2] $$

$C$ is somehow the $\bar{h}(x)$, the average of the outputs of the model, based on different training sets. So, we can write the error as:

$$ \mathbb{E}_D[A^2] = \mathbb{E}_D[(h_D(x)-\bar{h}(x))^2] $$

Now, let's focus on the second term. We can write it as:

$$ \mathbb{E}_D[B^2] = \mathbb{E}_D[(C-f(x))^2] $$

But since $C$ is the expectation of $h_D(x)$, we can write it as:

$$ \mathbb{E}_D[B^2] = \mathbb{E}_D[(\mathbb{E}_D[h_D(x)]-f(x))^2] $$

Remember! $C$ is the $\bar{h}(x)$, the average of the outputs of the model, based on different training sets. So, we can write the error as:

$$ \mathbb{E}_D[B^2] = \mathbb{E}_D[(\bar{h}(x)-f(x))^2] $$

Now, we can write the error as:

$$ 
\mathbb{E}_D[(h_D(x) - f(x))^2] = 
\underbrace{\mathbb{E}_D[(h_D(x)-\bar{h}(x))^2]}_{\text{Variance}} + 
\underbrace{\mathbb{E}_D[(\bar{h}(x)-f(x))^2]}_{\text{Bias}^2} 
$$

This is the decomposition of the error into two parts. The first part is the variance of the model, and the second part is the bias of the model. The variance of the model is the expected error of the model with respect to different training sets. The bias of the model is the expected error of the model with respect to different true functions. The bias-variance trade-off is the trade-off between these two errors. The goal is to find a model which has a low bias and a low variance. However, in practice, it is not always possible to find a model with both low bias and low variance. So, you have to choose a model which has a balance between bias and variance.

> Variance shows the average difference between $h_D(x)$ (the models output) and $\bar{h}(x)$ (the average of the models output). Bias shows the average difference between $\bar{h}(x)$ and $f(x)$ (the true function). In other words, variance shows how much the model is sensitive to the training set. Bias says no matter how hard you try, your average model will (probably) never be equal to the true relation $f$.

### Summary of the math section:

Assume there is a true underlying function \( f(x) \) that maps inputs to outputs. However, since this true function is unknown, we attempt to approximate it using a model \( h_D(x) \), trained on a dataset \( D \).

The expected error for a fixed input \( x \) is given by:

$$
\mathbb{E}_D[(h_D(x) - f(x))^2]
$$

This error can be decomposed into two key components: bias and variance. After applying some mathematical transformations, we arrive at:

$$
\mathbb{E}_D[(h_D(x) - f(x))^2] = 
\underbrace{\mathbb{E}_D[(h_D(x) - \bar{h}(x))^2]}_{\text{Variance}} + 
\underbrace{(\bar{h}(x) - f(x))^2}_{\text{Bias}^2}
$$

Here:
- **Variance**: $\mathbb{E}_D[(h_D(x) - \bar{h}(x))^2]$, the variability of the model's predictions $h_D(x)$ around the mean prediction $\bar{h}(x)$. This captures the model's sensitivity to changes in the training data.
- **Bias**: $(\bar{h}(x) - f(x))^2$, the difference between the average model prediction $\bar{h}(x)$ and the true function $f(x)$. It reflects how far the model's predictions are from the true relationship.

The total error is the sum of these two components. The **bias-variance trade-off** arises because reducing one often increases the other. For example:
- Increasing model complexity reduces bias but increases variance.
- Simplifying the model reduces variance but increases bias.

## Key Insights and Practical Implications

- **Bias** leads to underfitting: The model is too simplistic to capture the data's complexity.
- **Variance** leads to overfitting: The model is too sensitive to the training data and captures noise as if it were signal.
- Smaller datasets benefit from simpler models to avoid overfitting. Larger datasets can support more complex models without excessive variance.
- The goal is to strike a balance between bias and variance, finding a model that minimizes the total error.

> **Summary**: Bias measures how far the average prediction is from the true function, while variance measures how much predictions vary across different datasets. A good model strikes a balance, avoiding both underfitting and overfitting.

## Points to Remember

- **High bias** = Strong assumptions, simple models, underfitting.
- **High variance** = Complex models, overly flexible, overfitting.
- Balance is key: Neither extreme leads to optimal performance.
- The dataset size and problem complexity play significant roles in determining the ideal model complexity.

Understanding the bias-variance trade-off is crucial for building machine learning models that generalize well to unseen data, ensuring both accuracy and reliability.