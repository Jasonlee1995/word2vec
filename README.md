# Word2Vec Implementation with Pytorch


## 0. Develop Environment
- Docker Image : tensorflow/tensorflow:1.13.2-gpu-py3-jupyter
- Pytorch : Stable (1.5) - Linux - Python - CUDA (10.2)
- Using Multi-GPU (not tested on cpu only)


## 1. Brief Summary of *'Efficient Estimation of Word Representations in Vector Space'*
### 1.1 Goal
- Introduce techniques that can be used for learning high-quality word vectors from huge data sets with billions of words, and with millions of words in the vocabulary
- Maximize accuracy of vector operations by developing new model architectures that preserve the linear regularities among words

### 1.2. Intuition
- Most of the complexity is caused by the non-linear hidden layer in the model
- Explore simpler models that might not be able to represent the data as precisely as neural networks, but can possibly be trained on much more data efficiently

### 1.3. New Log-linear Models
![Figure 2](./Figures/Figure_01.png)
#### 1.3.1. Continuous Bag-of-Words Model (CBOW)
- Network : Projection layer + Average projected vectors + Output layer
- Predict current word based on the context
- Best performance by building a log-linear classifier with four future and four history
words at the input

#### 1.3.2. Continuous Skip-gram Model (Skip-gram)
- Network : Projection layer + Output Layer
- Predict surrounding words given the current word
- Select randomly a number R in range < 1 : C > where C is the maximum distance of words and use R words from history and R words for future of the current word as correct labels
- Increasing the range improves quality but it also increases the computational complexity
- Give less weight to the distant words by sampling less from those words since the more distant words are usually less related to the current
- Use C = 10 in paper

### 1.4. Train
- Epoch : 3
- Optimizer : Stochastic Gradient Descent (SGD) + backpropagation
- Learning rate : 0.025 (decrease linearly, so that it approaches zero at the end of the last training epoch)


## 2. Reference Paper
- Efficient Estimation of Word Representations in Vector Space [[paper]](https://arxiv.org/pdf/1301.3781.pdf)
