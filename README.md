# Dual-sourcing Inventory Problem

It is common practice that companies depend on multiple suppliers for product ordering. Suppose we have a regular supplier $R$, with a longer lead time $L_r$ and a lower cost $c_r$, and an express supplier $E$, with a shorter lead time $L_e$ and a higher cost $c_e$. Assume we have a sequence of i.i.d. demands $\{D_t, t \ge 1\}$, distributed as the nonnegative random variable $D$. Denote the unit holding and backorder costs by $h$ and $b$ respectively. Let $I_t$ denote the on-hand inventory, and  $\mathbf{q}_t^r =\{q_{t-i}^r,i\in[L_r]\}$, $\mathbf{q}_t^e =\{q_{t-i}^e,i\in[L_e]\}$ denote the pipeline vectors of orders placed but not yet delivered with $R$ and $E$ at the start of period $t$, where $q_{t-i}^r, q_{t-i}^e$ are the orders placed in period $t-i$.

At period $t$, a sequence of events happen in the following order:

- The on-hand inventory $I_t$ is observed.
- New orders $q_t^r$ and $q_t^e$ are placed with $R$ and $E$.
-  New inventory $q_{t-L_r}^r + q_{t-L_e}^e$ is delivered and added to the on-hand inventory.
- The demand $D_t$ is realized; the inventory and pipeline vectors are updated.
-  Costs for period $t$ are incurred.

Notice the on-hand inventory is updated according to 
$$
\begin{equation*}
    I_{t+1} = I_t + q_{t-L_r}^r + q_{t-L_e}^e - D_t.
\end{equation*}
$$
The pipeline vectors are updated according to
$$
\begin{align*}
    \mathbf{q}_{t+1}^r &= (q_{t-L_r+1}^r, \dots, q_{t-1}^r, q_t^r), \\
    \mathbf{q}_{t+1}^e &= (q_{t-L_e+1}^e, \dots, q_{t-1}^e, q_t^e).
\end{align*}
$$
Let $C_t$ be the sum of the ordering cost and holding and backorder costs incurred in time period t:
$$
\begin{equation*}
    C_{t} = c_r q_t^r + c_e q_t^e + h I_{t+1}^+ + b I_{t+1}^-.
\end{equation*}
$$
An admissible policy $\pi$ consists of a sequence of deterministic measurable functions $\{f_t^{\pi}, t\geq 1\}$ from $\mathbb{R}^{L_r + L_e + 1}$ to $\mathbb{R}^2_+$. Specifically, the new orders placed in period $t$ are given by $(q_t^r, q_t^e) = f_t^{\pi} (\mathbf{q}_t^r, \mathbf{q}_t^e, I_t)$. Let $\Pi$ denote the family of all admissible policies. The cost under a policy $\pi$ is denoted by $C_t^{\pi}$. We aim to minimize the long-run average cost
$$
\begin{equation*}
    C(\pi) = \limsup_{T \rightarrow \infty} \frac{1}{T} \sum_{t = 1}^{T} \mathbb{E}[C_t^{\pi}].
\end{equation*}
$$


Assume the demands follow Poisson distribution: $D \sim \mathrm{Pois}(\lambda)$, where $\lambda > 0$. Furthermore, assume the orders can only take integer values. Then the above process can be formulated as a discrete MDP. At period $t$, let $s_t = (\mathbf{q}_t^r, \mathbf{q}_t^e, I_t)$ be the state of the system, and let $a_t = (q_t^r, q_t^e)$ be the action taken. The state space $\mathcal{S}$ and action space $\mathcal{A}$ are given by
$$
\begin{equation*}
    \mathcal{S} = \mathbb{Z}_+^{L_r} \times \mathbb{Z}_+^{L_e} \times \mathbb{Z}, \quad \mathcal{A} = \mathbb{Z}_+^2.
\end{equation*}
$$
Note that $C_t$ is a function of $s_{t+1}$ instead of $s_t$. So the reward of step $t$ is actually received at step $t-1$:
$$
\begin{equation*}
    r(s_t, a_t) = - C_{t-1} = -c_r q_{t-1}^r - c_e q_{t-1}^e - h I_t^+ - b I_t^-.
\end{equation*}
$$
Define the function $g: \mathbb{R}^{L_r + L_e + 1} \times \mathbb{R}^{2} \rightarrow \mathbb{R}^{L_r + L_e + 1}$ as
$$
\begin{equation*}
    g(x_1, \dots, x_{L_r}, y_1, \dots, y_{L_e}, z, a_1, a_2) = (x_2, \dots, x_{L_r}, a_1, y_2, \dots, y_{L_e}, a_2, z + x_1 + y_1).
\end{equation*}
$$
Then
$$
\begin{equation*}
    s_{t+1} = g(s_t, a_t) - (0, \dots, 0, D_t).
\end{equation*}
$$
Hence the transition probabilities are given by
$$
\begin{equation*}
    \mathbb{P}(s_{t+1} = g(s_t, a_t) - (0, \dots, 0, k) \mid s_t, a_t) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, \dots.
\end{equation*}
$$


