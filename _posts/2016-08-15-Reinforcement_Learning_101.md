---
layout: post
title:  "Reinforcement Learning 101"
date:   2016-08-15 20:53:58 +0200
categories: reinforcement-learning machine-learning
---


## Reinforcement Learning 101 ##

**Reinforcement Learning** (RL) is a field of research on the study of **agents** that can **self-learn** how to behave through feedback, **reinforcement**, from its **environment**, a sequential decision problem. RL is a subfield of **Machine Learning**, which in turn is a subfield of **Artificial Intelligence or Computer Science**. 

RL differs from the more common machine learning task of **Supervised Learning**, wherein one is provided with a labeled dataset and has to infer a function that generalizes well on unseen examples. In RL, one starts without any such dataset and the agent has to obtain data through interaction in order to learn how to behave, and usually one intents to optimize the *accumulated reward over the agent's lifetime*.

For example, let's suppose we have a robot (our agent) that we send to Mars (the environment). We wish it to make a lot of distance and gather some rocks along the way. Although we could try to model the environment, it will require quite some resources and may be somewhat inaccurate. Controlling the robot from Earth is also not a good option, as our instructions will arrive minutes later. Thus, it might be a good idea to make our robot capable of learning by itself such that it can optimize its task.



## Modelling the environment ##

Although the idea of RL is to avoid the necessity of modelling an environment, it is naturally very useful to model some environments in order to research how agents would behave and learn when they start from scratch. 

In this blog post we will consider **Markov decision processes**. A Markov decision process (MDP) is an often-used discrete-time framework for modelling the dynamics of an environment ([Howard1960][Howard1960]; [Puterman1994][Puterman1994]). A very explicit notation for an MDP is a sextuple $$M = (\mathbb{S}, \mathbb{A}, P, R, I, \gamma)$$, where


- $$\mathbb{S}$$ is the state space; the set of states that the MDP can be in. A state contains information regarding the environment. E.g., when flying a helicopter, a state pertains to the position and orientation of the helicopter. Let $$s \in S$$ denote a context-dependent state, and let $$s_t$$ denote the observed state at timestep $$t$$.
- $$\mathbb{A}$$ is the action space; the set of actions available to the agent. E.g., the pilot can choose to yaw, roll or pitch the helicopter. Let $$a$$ denote a context-dependent action, and let $$a_t$$ denote the action taken at $$t$$.
- $$P : \mathbb{S} \times \mathbb{A} \times \mathbb{S} \to [0, 1]$$; is a transition function that outputs a probability for a transition to a successor state after taking an action in a state. Let $$S_t$$ denote the random variable for the state at $$t$$, and $$s'$$ denote the immediate subsequent state. For each state $$s$$ and action $$a$$, the probability of transitioning to a successor state $$s'$$ respects constraints $$Pr(s,a,s') \geq 0$$ and $$\sum_{s \in S} Pr(s,a,s') = 1$$. E.g., after steering, the helicopter makes a turn.
- $$R : \mathbb{S} \times \mathbb{A} \times \mathbb{S} \times \mathbb{R} \to [0, 1]$$; is a reward function that outputs a probability for a numeric reward given a transition. Let $$R_t$$ denote the random variable for the reward at $$t$$, and let $$r_t$$ denote the observed reward at $$t$$. E.g., arriving at the destination incurs a high reward, and crashing incurs a negative reward.
- $$I : \mathbb{S} \to [0, 1]$$; is an initial state function that outputs the probability that a state is the MDPs initial state $$s_0$$. E.g., the helicopter can take off at several locations.
- $$\gamma \in [0, 1]$$; is a discount rate parameter that weighs off imminent rewards versus long-term rewards.

In the literature, one may find different notations. Commonly, no initial state function is mentioned, or the transition and reward functions are defined differently (or even notated as T). To the best of my knowledge, the above notation is the most explicit. Actual implementations of MDPs employ datastructures of the programming language. For example, a numpy matrix with values between 0 and 1 describe the transition of a state-action pair (row) to a subsequent state (column).


At the beginning of interaction with an MDP, an agent observes an initial state $$s_0$$ and chooses an action $$a_0$$. From that point on, at each discrete timestep $$t = 0, 1, 2, \dots$$, it observes a successor state $$s_{t+1}$$ and a reward $$r_{t+1}$$, and chooses a new action accordingly.


![Agent - Environment interaction]({{ site.baseurl }}/images/agentenvironment.png)




## Online / Offline Reinforcement Learning ##




## References
[Howard1960]: R.A. Howard. Dynamic programming and Markov processes. MIT Press, 1960.
[Puterman1994]: M.L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, Inc. new York, NY, USA, 1994.

