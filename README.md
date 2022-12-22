# MNIST Generative Adversarial Networks

![AI](https://user-images.githubusercontent.com/29410712/209227614-8bae7ea8-825a-4320-a4ca-21b879ab9d1b.jpg)

Artificial intelligence (AI) is the simulation of human intelligence processes by machines. AI is divided into three stages: Artificial Narrow Intelligence (ANI), Artificial General Intelligence (AGI), and Artificial Super Intelligence (ASI). ANI involves machine learning and specializes in one area, AGI refers to a machine's ability to understand or learn any intellectual task similar to a human, and ASI entails having the intellectual powers beyond any humans. Currently, superintelligence is hypothetical and the technology does not exist. In this analysis, we we will breakdown the components of the Generative Adversarial Networks (GANs) and generate images using the MNIST dataset. The purpose is to introduce the idea of using deep neural networks for artificial intelligence. 

## Artificial Neural Networks
Neural networks (also known as artificial neural networks, or ANN) are a set of algorithms that are modeled after the human brain and is categorized as Artificial General Intelligence. Neural networks are an advanced form of machine learning that recognizes patterns and features in input data and provides a clear quantitative output. In its simplest form, a neural network contains layers of neurons, which perform individual computations. These computations are connected and weighed against one another until the neurons reach the final layer, which returns a numerical result, or an encoded categorical result.

One way to use a neural network model is to create a classification algorithm that determines if an input belongs in one category versus another. Alternatively neural network models can behave like a regression model, where a dependent output variable can be predicted from independent input variables. Therefore, neural network models can be an alternative to many of the models we have learned throughout the course, such as random forest, logistic regression, or multiple linear regression.

There are a number of advantages to using a neural network instead of a traditional statistical or machine learning model. For instance, neural networks are effective at detecting complex, nonlinear relationships. Additionally, neural networks have greater tolerance for messy data and can learn to ignore noisy characteristics in data. The two biggest disadvantages to using a neural network model are that the layers of neurons are often too complex to dissect and understand (creating a black box problem), and neural networks are prone to overfitting (characterizing the training data so well that it does not generalize to test data effectively). However, both of the disadvantages can be mitigated and accounted for.

Conceptually, neural networks involve multi-dimensional linear equations and dot products. To simplify the explanation, we will use the **Rosenblatt perceptron model**. The Rosenblatt perceptron model is a binary single neural network unit, and it mimics a biological neuron by receiving input data, weighing the information, and producing a clear output.

The Rosenblatt perceptron model has four major components:

+ Input values, typically labelled as $x$ or 𝝌 (chi, pronounced kaai, as in eye)
+ A weight coefficient for each input value, typically labelled as $w$ or ⍵ (omega)
+ Bias is a constant value added to the input to influence the final decision, typically labelled as $w_0$.
+ A net summary function that aggregates all weighted inputs, in this case a weighted summation:

![data-19-1-2-1-weighted-summation](https://user-images.githubusercontent.com/29410712/208508271-b775459f-f379-42a5-9c54-6d8877791f0a.png)

The inputs values $(x_1, x_2, \cdots, x_n)$ are multiplied with the corresponding weights $(w_1, w_2, \cdots, w_n)$ to produce the output $y$. As illustrated in the larger circle, we sum the weighted inputs to obtain the total amount of input which includes the bias term $w_0$. The equation can be rewritten as: 

$$
  y = \sum_{i=1}^n w_ix_i + w_0
$$

$$
  y = w_1x_1 + w_2x_2 + \cdots + w_ix_i + w_0
$$

As we can see, the Rosanblatt perceptron model is just a multilinear regression model.

## Generative Adversarial Networks (GANs)
