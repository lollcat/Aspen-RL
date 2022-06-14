# SAC Agent
We base our sac implementation of acme's jax sac implementation, with adjustments to account for 
weirdness in our environment. We also use our own training script / buffer 
(rather than acme's environment loop and reverb). 

## learning.py
We use the _step function from acme's learner.
Additionally we adjust the "next observation" to include a discount corresponding to the tops and 
bottoms streams, so that the value of the next state (s') is as follows
$ V(s') = \gamma_{tops} V(s'_{tops}) + \gamma_{bots} V(s'_{bots}) $
where $V(s'_{tops})$ and $ V(s'_{bots}) $ are calculated as forwarded passes through the value 
network. We encode this difference between the "observation" and "next observation" into the 
networks definition, such that we do not have to edit the sac learner. 

## networks.py
Define the signatures of the actor and critic networks. Follows acme's networks.py file for sac, 
but with more explicit type hinting.

## create_networks.py
Create the sac networks (to meet the definition in networks.py) using haiku.

## agent.py
Define the agent which is composed of :
 - a select action function
 - an update function, which updates the trainable parameters of the agent.

## buffer.py
A replay buffer where we store experience generated within the environment, for training the agent.

## train.py
A training script that uses the above building blocks to train a sac agent.