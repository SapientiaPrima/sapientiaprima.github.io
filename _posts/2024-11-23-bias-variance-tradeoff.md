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

First, let us fix some notations. You have a hypothesis set $H$, which is basically the set of all functions you can accept as your final model. Note that your basic assumption is that there is a function $f$ (the true function) which represents the relation between features and labels. For a classification problem, it means that $f$ can take features of the training samples and map them to the labels of the corresponding samples. 
