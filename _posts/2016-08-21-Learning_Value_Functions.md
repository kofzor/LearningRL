---
layout: post
title:  "Learning Value Functions"
date:   2016-08-21 20:55:58 +0200
categories: reinforcement-learning machine-learning
comments: true
---


## Value Functions ##

In this blog post we gonna talk about one of my favorite parts of reinforcement learning: **Value Functions**.

A *value function* maps each state to a value that corresponds with the output of the objective function. We use $$V_{M}^{\pi}$$ to denote the value function when following $$\pi$$ on $$M$$, and let $$V_{M}^{\pi}(s)$$ denote the *state-value* of a state $$s$$. In the literature $$M$$ is often omitted as it is clear from context. A state-value for the expected return of a policy on $$M$$ is defined by

$$ V_{M}^{\pi}(s) = \mathbb{E}_{\pi, M} \{ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \vert S_t = s \}

where $$\mathbb{E}_{\pi, M} \{ X \vert Y \}$$ is the conditional expectation of $$X$$ given $$Y$$ when following $$\pi$$ on $$M$$. Thus, a state-value indicates how good it is to be in its respective state in terms of the expected return. If we want to perform well, we prefer states with high state-values.

The value function of the optimal policy, $$\pi^*$$ is called the optimal value function and is denoted by $$V_{M}^{\pi^*}$$ and in the literature usually just $$V^*$$. Evidently, it holds that for each state, $$V^*(s)$$ is equal or higher than for other policies:

$$ \forall_{\pi \in \Pi, s \in \matbb{S}} : V^*(s) \geq V^{\pi}(s)$$



## Q-functions ##

An *action-value function* or more commonly known as *Q-function* is a simple extension of the above that also accounts for actions. It is used to map combinations of states and actions to values. A single combination is often referred to as a *state-action pair*, and its value as a (policy) *action-value*.

We use $$Q_{M}^{\pi}$$ to denote the Q-function when following $$\pi$$ on $$M$$, and let $$Q_{M}^{\pi}(s,a)$$ denote the *action-value* of a state-action pair $$(s,a)$$. In the literature, it is common to leave out both $$\pi$$ and $$M$$. The action-value is then:

$$ Q_{M}^{\pi}(s,a) = \mathbb{E}_{\pi, M} \{ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \vert S_t = s, A_t = a \}

which corresponds to the idea that when you are in state $$s$$ and take $$a$$ and follow $$\pi$$ afterwards then the expectation is as above.

The Q-function of $$\pi^*$$ is called the optimal Q-function. It is usually noted as $$Q^*$$. Notice that state-values and action-values are related to each other as follows:

$$V^{\pi}(s) = \sum_{a\in\mathbb{A}}\pi(s,a)Q^{\pi}(s, a)$$ and $$V^*(s) = max_{a\in\mathbb{A}}Q^*(s, a)$$ 



### Deriving policies from Q-functions ###

It is very straightforward to derive policies from Q-functions as they describe how good each action is. In the literature, the most common derivation is the *greedy policy* of a Q-function: in each state, choose the *greedy action*, which is the action with the highest action-value. Notice that if we know $$Q^*$$ then we can find $$\pi^*$$ by *acting greedy with respect to $$Q^*$$*.

Q-functions are frequently used to guide exploration and exploitation. The most common approach is to use **epsilon-greedy**: at each timestep, choose a greedy action with probability $$1-\epsilon$$ or choose a random action with probability $$\epsilon$$ [Sutton and Barto, 1998]. Interestingly, in the literature never considers the case that a random action might still choose the same greedy action and hence the chance of choosing the greedy action is usually higher than $$1-\epsilon$$. When using epsilon-greedy, one commonly starts with a high $$\epsilon$$ and decrease it over a time. Evidently, when $$\epsilon = 0$$, it is equal to the greedy policy.

Another approach I like to mention is *Boltzmann exploration*, where one introduces a *temperature parameter* $$\beta$$ to map action-values to action probabilities as follows:

$$ Pr(a) = \frac{exp(Q(s,a) / \beta)}{\sum_{b \in \mathbb{A}} exp(Q(s,a) / \beta)}

The parameter is used to control how the difference in action-values corresponds to a difference in action-probabilities. As $$\beta$$ goes to zero, Boltzmann chooses greedily, and as $$\beta$$ goes to infinity, all actions have an equal chance. In different research fields this formula is also known as *softmax*.











## References
[Sutton and Barto, 1998] R.S. Sutton and A.G. Barto. Reinforcement Learning: An Introduction (Adaptive Computation and Machine Learning). The MIT Press, 1998. 

[C. Szepesvari, 2010] Algorithms for Reinforcement Learning. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool, 2010.

[H. van Hasselt. 2011] Insights in Reinforcement Learning. PhD thesis, Utrecht University, 2011.