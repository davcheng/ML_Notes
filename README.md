## One MBAs quest to learn Machine Learning like his (startup) life depends on it.
The barrier to entry for getting into ML is pretty steep, and not because it's tough, but because a lot of the resources are jargon heavy and throw in complex formulas to scare non-nerds away. This is an attempt to try and simplify everything I've learned - and am continuing to learn - around ML, neural networks, deep learning, etc.

# What is Machine Learning vs. What isn't Machine learning
![alt text](https://imgs.xkcd.com/comics/machine_learning.png)  
Machine learning, at its very core, is learning to predict an outcome based on a large set of examples. For instance, if you are trying to find out if a picture is pizza or not, you could show it a bunch of pictures of pizza and a bunch of pictures of "not pizza" and slowly (after training), the model will figure out how to tell the difference (very similar to how you would teach a child) based on features it is identifying. For instance, just like a child, the model could start to identify that triangular shapes are common in all of the "TRUE, THIS IS A PIZZA (slice)" pictures and use that as a hint.

Conversely, a Non-Machine Learning approach would be, "I know what pizza looks like, I'm going to write code that looks for triangles that have this proportion, and look for pepperonis on it" - the code would NEVER need to see a pizza first, it would just work. What you're doing here is "hard-coding the model" yourself so that you don't need to train it with a lot of data. For instance, if you need to teach someone what pizza is but have no examples to show them, you would just describe it and say "remember this if you ever get to a critical "is this pizza?" moment. And again, this Non-ML approach can be great for somewhat easy (read: standardized) tasks like "Pizza or Not", but breaks down for more complex tasks/if you don't, a. have incredible domain knowledge of pizza, b. feel like coding every possible feature that could discern between pizza or not pizza.

Now if all pizzas were pretty similar, and you are okay with a fairly rigid definition (perhaps you are just at only Cici's pizzas, and only accept cici's cheese pizza as the only true pizza), it might be far less efficient to use a machine learning approach since you already know exactly what you're looking for (white isosceles triangle, between 4-5 inches with a brown rounded edge).

As you can see, ML can be pretty powerful but if you know all of the features, expect a relatively consistent set of input data that you need to predict, and don't have many pictures to train things on, you are probably better off

# Overview
There are three types of learning:
1. Supervised Learning - learning to predict an output when given an input vector. There are two types: Regression (output you're guessing is a real number, e.g., price of stocks in six months), and Classification (output is a class label, e.g., is this a hot dog?)
2. Unsupervised Learning - Discover good internal representation of an input
3. Reinforcement Learning - Learning to select an action (decision) based on maximizing some payoff function (used for Alpha Go in teaching it how to play Go)



# 1. Basic principle
y = Wx+b
where y = output, x = input, W = weights (aka parameters that are slowly tuned) and b = bias

## Linear regression:
y = Wx + Wx + ... + Wx +b
This maps input value(s) to a numerical output; e.g., given a person's weight, height, and shoe size, can we guess their IQ?
(pretend there is a weird correlation where fat, tall, big-footed people are smarter).

## Logistic (or Logit) regression:
just like Linear regression but used for categorical data;
for most ML resources, logistic regression refers to binary categorical data (true/false, yes/no, red/blue). However, can also create
multinomial logistic regression for classifying data with 3 or more categories.

The architecture for classification looks like this:
1. Net input function: take inputs and pump them through weights and bias
2. Activation function: convert the ouput to a normalized scale ([0,1] for some activation functions like tanh or logistic sigmoid, [0,max x] for others like ReLU)
3. Unit Step function: take the output of the activation function and create a decision rule (if >.5, classify as red; if <=.5, classify as blue);

notes: this is still linear because the decision surface is still linear.

# Glossary

## Terms Related to Model Architecture:

### Vanishing gradient -
Some training methods are gradient-based, meaning, they play around with weights (parameters) and see how that affects the output (whatever the model is trying to guess). Since the learning is computed based on how much a change to a parameter (i.e., if I see blue, I will more likely think this is a picture of an ocean) impacts a change in the output (is this a picture of an ocean?), if the gradient becomes too small, the network will have a very difficult time learning because the levers that it's trying to pull for guess and check are way too small (i.e. a large change in the parameter value doesn't have a big impact on the output). All of this is dependent on the activation function as each layer "squashes" input into a small output range (sigmoid maps ALL possible numbers to [0,1]), and after several layers of this squashing, there is hardly any change in the output. Effectively making it so that if I show a picture of an ocean with all this blue, the model won't really know because the features it's looking for haven't been clearly identified.

problem is exacerbated as the number of layers increases.
https://www.quora.com/What-is-the-vanishing-gradient-problem


### Activation Functions -
Used to go from a

### ReLU - Rectified Linear Unit
This is an activation function


### Dropout - ignoring random nodes so that
