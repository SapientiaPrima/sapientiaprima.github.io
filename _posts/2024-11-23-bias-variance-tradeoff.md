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