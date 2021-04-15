# Dual-sourcing Inventory Problem

It is common practice that companies depend on multiple suppliers for product ordering. 
Suppose we have a regular supplier <img src="https://latex.codecogs.com/gif.latex?R" />, with a longer lead time <img src="https://latex.codecogs.com/gif.latex?L_r" /> and a lower cost <img src="https://latex.codecogs.com/gif.latex?c_r" />, and an express supplier <img src="https://latex.codecogs.com/gif.latex?E" />, with a shorter lead time <img src="https://latex.codecogs.com/gif.latex?L_e" /> and a higher cost <img src="https://latex.codecogs.com/gif.latex?c_e" />. Assume we have a sequence of i.i.d. demands <img src="https://latex.codecogs.com/gif.latex?\{D_t,t\ge1\}" />, distributed as the nonnegative random variable <img src="https://latex.codecogs.com/gif.latex?D" />. Denote the unit holding and backorder costs by <img src="https://latex.codecogs.com/gif.latex?h" />and <img src="https://latex.codecogs.com/gif.latex?b" /> respectively. Let <img src="https://latex.codecogs.com/gif.latex?I_t" /> denote the on-hand inventory, and <img src="https://latex.codecogs.com/gif.latex?\mathbf{q}_t^r=\{q_{t-i}^r,i\in[L_r]\},\mathbf{q}_t^e=\{q_{t-i}^e,i\in[L_e]\} " /> , denote the pipeline vectors of orders placed but not yet delivered with <img src="https://latex.codecogs.com/gif.latex?R" /> and <img src="https://latex.codecogs.com/gif.latex?E" /> at the start of period <img src="https://latex.codecogs.com/gif.latex?t" />, where <img src="https://latex.codecogs.com/gif.latex?q_{t-i}^r,q_{t-i}^e" /> are the orders placed in period <img src="https://latex.codecogs.com/gif.latex?t-i" />.

At period $t$, a sequence of events happen in the following order:

- The on-hand inventory <img src="https://latex.codecogs.com/gif.latex?I_t" /> is observed.
- New orders <img src="https://latex.codecogs.com/gif.latex?q_t^r" /> and <img src="https://latex.codecogs.com/gif.latex?q_t^e" />  are placed with <img src="https://latex.codecogs.com/gif.latex?R" /> and <img src="https://latex.codecogs.com/gif.latex?E" />.
-  New inventory <img src="https://latex.codecogs.com/gif.latex?q_{t-L_r}^r+q_{t-L_e}^e" /> is delivered and added to the on-hand inventory.
- The demand <img src="https://latex.codecogs.com/gif.latex?D_t" /> is realized; the inventory and pipeline vectors are updated.
-  Costs for period <img src="https://latex.codecogs.com/gif.latex?t" /> are incurred.

Notice the on-hand inventory is updated according to 
<img src="./img/1.png" style="float: center;" />

The pipeline vectors are updated according to

<img src="./img/2.png" style="float: center;" />

Let <img src="https://latex.codecogs.com/gif.latex?C_t" /> be the sum of the ordering cost and holding and backorder costs incurred in time period t:

<img src="./img/3.png" style="float: center;" />

An admissible policy <img src="https://latex.codecogs.com/gif.latex?\pi" />  consists of a sequence of deterministic measurable functions <img src="https://latex.codecogs.com/gif.latex?\{f_t^{\pi},t\ge1\}" /> from <img src="https://latex.codecogs.com/gif.latex?\mathbb{R}^{L_r+L_e+1}" /> to <img src="https://latex.codecogs.com/gif.latex?\mathbb{R}^2_+" />. Specifically, the new orders placed in period <img src="https://latex.codecogs.com/gif.latex?t" /> are given by <img src="https://latex.codecogs.com/gif.latex?(q_t^r,q_t^e)=f_t^{\pi}(\mathbf{q}_t^r,\mathbf{q}_t^e,I_t)" />. Let <img src="https://latex.codecogs.com/gif.latex?\Pi" /> denote the family of all admissible policies. The cost under a policy <img src="https://latex.codecogs.com/gif.latex?\pi" />  is denoted by <img src="https://latex.codecogs.com/gif.latex?C_t^{\pi}" />. We aim to minimize the long-run average cost

<img src="./img/4.png" style="float: center;" />

Assume the demands follow Poisson distribution: <img src="https://latex.codecogs.com/gif.latex?D\sim\mathrm{Pois}(\lambda)" /> , where <img src="https://latex.codecogs.com/gif.latex?\lambda>0" />. Furthermore, assume the orders can only take integer values. Then the above process can be formulated as a discrete MDP. At period  <img src="https://latex.codecogs.com/gif.latex?t" />, let  <img src="https://latex.codecogs.com/gif.latex?s_t=(\mathbf{q}_t^r,\mathbf{q}_t^e,I_t)" /> be the state of the system, and let <img src="https://latex.codecogs.com/gif.latex?a_t=(q_t^r,q_t^e)" /> be the action taken. The state space <img src="https://latex.codecogs.com/gif.latex?\mathcal{S}" /> and action space <img src="https://latex.codecogs.com/gif.latex?\mathcal{A}" /> are given by

<img src="./img/5.png" style="float: center;" />

Note that <img src="https://latex.codecogs.com/gif.latex?C_t" /> is a function of <img src="https://latex.codecogs.com/gif.latex?s_{t+1}" /> instead of <img src="https://latex.codecogs.com/gif.latex?s_{t}" /> . So the reward of step <img src="https://latex.codecogs.com/gif.latex?t" />  is actually received at step <img src="https://latex.codecogs.com/gif.latex?t-1" /> :

<img src="./img/6.png" style="float: center;" />

Define the function <img src="https://latex.codecogs.com/gif.latex?g:\mathbb{R}^{L_r+L_e+1}\times\mathbb{R}^{2}\rightarrow\mathbb{R}^{L_r+L_e+1}" />  as

<img src="./img/7.png" style="float: center;" />

Then

<img src="./img/8.png" style="float: center;" />

Hence the transition probabilities are given by
<img src="./img/9.png" style="float: center;" />

