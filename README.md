## One MBAs quest to learn Machine Learning like his (startup) life depends on it.
The barrier to entry for getting into ML is pretty steep, and not because it's tough, but because a lot of the resources are so heavy with jargon. This is an attempt to try and simplify everything I've learned and am continuing to learn around ML, neural networks, deep learning, etc.

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
Some training methods are gradient-based, meaning, they play around with weights (parameters) and
see how that affects the ouput. Since the learning is computed based on how much a change in the parameter
impacts a change in the output (gradient), if the gradient becomes too small, the network will have a very difficult
time learning because the levers that it's trying to pull for guess and check are way too small (i.e. a large change in the
parameter value doesn't have a big impact on the output). All of this is dependent on the activation function as each layer "squashes"
input into a small output range (sigmoid maps ALL possible numbers to [0,1]), and after several layers of this squashing, there is hardly any change in the output.

problem is exacerbated as the number of layers increases.
https://www.quora.com/What-is-the-vanishing-gradient-problem


### Activation Functions -
Used to go from a

### ReLU - Rectified Linear Unit
This is an activation function


### Dropout - ignoring random nodes so that
