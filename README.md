## One MBAs quest to learn Machine Learning like his (startup) life depends on it.
The barrier to entry for getting into ML is pretty steep, and not because it's tough, but because a lot of the resources are jargon-heavy and throw in complex formulas to intimidate. This is an attempt to try and simplify everything I've learned - and am continuing to learn - around ML, neural networks, deep learning, etc.

# What is Machine Learning vs. What isn't Machine learning
![alt text](https://imgs.xkcd.com/comics/machine_learning.png)  
>"In machine learning you use data to train a model. That model will be used to make predictions on unseen data." - someone on Quora

Machine learning, at its core, is learning to predict an outcome based on a set of examples. For instance, if you are trying to figure out if a picture is "pizza" or "not". You could look through a bunch of pictures of "pizza" and a bunch of pictures of "not pizza", and slowly [or in machine learning speak, after "training"], you would figure out how to tell the difference.  Just like how a child (or any individual) who has never seen or heard of the concept of pizza, the model would (at some point) start to identify patterns, known as features, in determining what IS and what ISN'T pizza (e.g., triangular-shaped slices, round greasy-looking red objects (pepperonis), brown crust, etc.). These features would be remembered and stored to make future predictions.

Conversely, a Non-Machine Learning approach would be, "I know what pizza looks like, I'm going to write code that looks for triangles that have this proportion, and look for round pepperonis on it." The algorithm (model) would NEVER need to see a pizza first, it would just work. What you're doing here is "hard-coding the model" yourself so that you don't need to train it with a lot of data. For this stupid pizza example, the Non-ML approach would be: if instead, you needed to teach someone what pizza is, but have no examples to show them, you could just describe defining characteristics of pizza and say "remember this if you ever get to a critical, 'is this pizza?' moment". And again, this Non-ML approach can be great for many tasks of "Pizza or Not", but it can break down for more complex definitions of pizza/if you don't, a. have incredible domain knowledge of pizza, b. have incredible domain knowledge of what isn't pizza, or c. feel like coding every possible variant of features that could differentiate the two to create a robust model.

Now if all pizzas were pretty similar, and you are okay with a fairly rigid definition (perhaps you are just identifying only Domino's pizzas, and only accept a hand-tossed cheese pizza as the only true pizza), it might be far less efficient to use a machine learning approach since you already know exactly what you're looking for (white isosceles triangle, between 4-5 inches with a brown rounded edge, red marina sauce scattered throughout). But you'd be pretty doomed if you took your round algorithm to a place that serves grandma slices, where square-shaped pizzas would ruin you.

As you can see, ML can be pretty powerful but it's important to remember that it is very expensive (computationally). ML takes a pretty powerful computer/server to run on efficiently [not several days while you hope your laptop doesn't explode while you're sleeping at night]. So if you know all of the features, expect a relatively consistent set of samples to predict, and don't have a lot of data to train things on, you might be better off just keeping it simple with other "shallow" learning techniques (linear regression, K-Nearest Neighbor (kNN), Support Vector Machines (SVM), etc.).

# Overview
Now that you understand what the basic premise behind Machine Learning is, let's get into the three primary types of "learning":

1. Supervised Learning - When you have a bunch of data to be used for training your model that's been labeled. Labeled data means somewhere, somehow, each data row that you're using to train your model has been labeled with a "truth" (e.g., this row of data is pizza). Within supervised learning, there are two types: Regression (output you're guessing is a real number, e.g., price of stocks in six months) and Classification (output is a class label, e.g., is this pizza or not? is it fall, spring, summer, or winter based on temperature, precipitation, and humidity?)

2. Unsupervised Learning - Here's some data, I won't give you the right answer for the data - find interesting structure/patterns/groupings. For instance, if you give your algorithm a ton of pictures of pizza and not pizza, but don't tell it which one is which, it cannot magically determine that this is pizza, and this isn't, but it can cluster (group) the data into piles based on similarities (these inputs look/sound/feel similar). From there, you can extract what you need out of the groups. This is great for data exploration and finding patterns you didn't even know existed.

3. Reinforcement Learning - Learning to select an action (decision) based on maximizing some payoff function (used for Alpha Go in teaching it how to play Go).


# Basic mathematical principle
While training your model, there are 4 primary components:
1. the input (data)
2. the output (prediction)
3. weights (magic numbers, known as parameters, that are tweaked over and over to make your neural network -- or the math equation that turns your input into the desired output)
4. bias ()
y = Wx+b
where y = output, x = input, W = weights (the parameters that are slowly tuned to build your neural network) and b = bias (perhaps in the universe of pictures, there are more "not pizza" than "pizza", so if it's a toss up based on your trained weights (W), guessing "not pizza" would be prudent because there is a greater likelihood (bias)).

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

# Training
When training your model, you are in essence pushing a large quantity of data through a set of vectors (that represent weights), evaluating how correct your weights are (via a loss/cost function), then fine tuning the weights over and over until your data starts falling into the right categorization/predictions.

I think the best example of this is one of those mechanical coin sorters that you can feed coins into. Imagine it starts out randomly putting coins into buckets, after every few coins, it looks at how many it got right, and then updates it's decision criteria - eventually, it will change it's filters to minimize loss (incorrectly sorted coins) such that the next time it get's a coin, it can determine if it's a penny, nickel, dime, or quarter.

Training Set  --> Learning algorithm --> Prediction

## Gradient descent



# Performance
After you've built and trained your model, the next step is to figure out if it's actually any good. While it sounds pretty simple, it gets pretty sticky pretty quickly - particularly with these three terms - accuracy, precision, and recall - that I am convinced no one truly remembers for more than 10 minutes. Hopefully this (and whatever diagrams your 7th grade science teacher showed you) will help.

Accuracy - This is the number of correct predictions divided by the number of total predictions. While this seems like a pretty good measure of performance (and it might be fine most of the time), data is tricky and horrible. For example, pretend you are screening people for some rare disease that only 1 in a 1000 people have. If your "algorithm" for determining if someone is "disease-free" is "EVERYONE IS DISEASE-FREE", your accuracy would be pretty stellar (99.9% to be exact). And if your boss only used accuracy to evaluate performance, he'd probably give you a promotion. But, unfortunately, you're probably going to be fired because your algorithm literally didn't identify a single person who actually had the disease. That's why accuracy isn't sufficient, and that's where precision and recall come in.

Precision - This is kind of like your "catch-rate". In the disease prediction example, precision is "how many people did you classify as having the disease", divided by the "total number of people that had the disease". Meaning, if you screened 3000 people, and 3 had the disease, and you're algorithm correctly identified 2 of them as having the disease, your precision would be 2/3. If in 3000 people, 3 had the disease, and you guessed 2999 of them had the disease and missed one of the ones that didn't, you're precision would still be 2/3 (since it doesn't care about the ones you marked as over-marked as "diseased"). There are certain cases where you would want higher precision, which would be more conservative (for instance, cancer screenings), where as other cases, you would wouldn't (spam emails... if only gmail would stop marking those important emails that I don't respond to because they were "marked as spam"...)

Recall - This the "how many good eggs do you have in your basket" rate. This is the number of people who actually had the disease, within all of the people you flagged as having a disease. 

Example Illustration
You have an algorithm that predicts whether or not a person is pregnant (perhaps based on shopping history). Let's assume 1000 people visit the site, and only 10 are pregnant:

Case 1: You predict no one is pregnant
accuracy = total correct predictions / total predictions = 990/1000 = 90%
precision = total correctly identified / total that could have been correctly identified = 0/10 = 0%
recall = total that were pregnant / total identified as pregnant = 0/0 = undefined

Case 2: You predict everyone is pregnant
accuracy = 10/1000 = 1%
precision = 10/10 = 100% (you got everyone that was pregnant)
recall = 10/1000 = 1%

Case 3: You

For something like, marking someone as having a rare, but treatable life-threatening disease, you would probably want an algorithm with better precision (don't let anyone slip through the cracks).

However, if you were making a spam detection algorithm for email filtering, you probably don't want to over mark things as spam (or you might miss an important email). In this example

These examples highlight the core principle in performance which is, in order to figure out if a model/algorithm is "good" or not, it is imperative to first understand the cost of mistakes (is it better to over guess or under guess?). Usually what people will do is assign a cost to each of the four scenarios (incorrectly saying someone IS pregnant, incorrectly saying some IS NOT pregnant, correctly saying someone IS pregnant, and correctly saying someone IS NOT pregnant).

## Flavors of NN

### Convolutional Neural Network
Image recognition

### Recurrent Neural Network

# Glossary

## Terms Related to Training

### Epochs vs. Batch Size vs. Iterations
Epoch - Pushing an entire set of training set (forward AND backward) through a neural network. Due to the nature of using gradient descent to update the weights, one epoch is generally not enough because the iterations gradually tune the weights (based on derivative of loss). However, too many epochs may result in overfitting as the model will be able to exactly guess data points from the training set, but not necessarily generalize to newer samples.

since one Epoch is generally too large to pass through at once you chunk the training data up into "batches"...  

Batch Size - total number of training examples in a batch; the larger the size, the more memory you'll need.

Iterations - total number of batches needed to complete an epoch (in other words, if you have 1000 training examples, and your batch size is 200, you'll have 5 iterations to complete an epoch)

Learning rate -

Gradient Descent -

## Terms Related to Model Architecture:

### Vanishing gradient -
Some training methods are gradient-based, meaning, they play around with weights (parameters) and see how that affects the output (whatever the model is trying to guess). Since the learning is computed based on how much a change to a parameter (i.e., if I see blue, I will more likely think this is a picture of an ocean) impacts a change in the output (is this a picture of an ocean?), if the gradient becomes too small, the network will have a very difficult time learning because the levers that it's trying to pull for guess and check are way too small (i.e. a large change in the parameter value doesn't have a big impact on the output). All of this is dependent on the activation function as each layer "squashes" input into a small output range (sigmoid maps ALL possible numbers to [0,1]), and after several layers of this squashing, there is hardly any change in the output. Effectively making it so that if I show a picture of an ocean with all this blue, the model won't really know because the features it's looking for haven't been clearly identified.

problem is exacerbated as the number of layers increases.
https://www.quora.com/What-is-the-vanishing-gradient-problem


### Activation Functions -
Used to go from a

### ReLU - Rectified Linear Unit
This is an activation function
https://www.quora.com/What-is-the-role-of-the-activation-function-in-a-neural-network-How-does-this-function-in-a-human-neural-network-system

### Dropout - ignoring random nodes so that

### One-hot Encoding
Taking categorical data and converting into numerical data using vectors (dummy variables).
If you want to indicate that something is summer versus winter versus fall


## Validation

### Confusion Matrix


Sources:
https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
