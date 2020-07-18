# model-free-rl-algos
=====================

This repository contains source codes for the paper titled _Model-free Reinforcement Learning in Infinite-horizon Average-reward Markov Decision Processes_ authored by
Chen-Yu Wei, Mehdi Jafarnia-Jahromi, Haipeng Luo, Hiteshi Sharma, Rahul Jain. 
The paper was accepted at [ICML 2020](https://icml.cc/Conferences/2020).

This paper proposes two model-free algorithms for tabular MDPs. The first algorithm _Optimistic Discounted Q-learning_ achieves a regret bound of _O(T<sup>2/3</sup>)_ under the assumption that the MDP is weakly-communicating; the second algorithm _MDP-OOMD_ achieves a regret bound of _O(T<sup>1/2</sup>)_ under the assumption that the MDP is ergodic. 

The codes are jointly implemented by Mehdi Jafarnia-Jahromi and Chen-Yu Wei. 
