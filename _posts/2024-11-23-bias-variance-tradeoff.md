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

## The Math Behind Bias and Variance Trade-off

First, let us fix some notations. You have a hypothesis set $H$, which is basically the set of all functions you can accept as your final model. Note that your basic assumption is that there is a function $f$ (the true function) which represents the relation between features and labels. For a classification problem, it means that $f$ can take features of the training samples and map them to the labels of the corresponding samples. In other words, $f$ is the best function you're looking for. However, unfortunately, you don't know $f$. Instead, you have the outputs given by it for some specific inputs, and you call inputs and outputs together "the training set". Denote this set by $D$.

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