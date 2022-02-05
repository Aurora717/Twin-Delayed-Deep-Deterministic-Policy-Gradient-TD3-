# Twin-Delayed-Deep-Deterministic-Policy-Gradient-TD3-

TD3 represents a culmination of a series of improvements as it is in essence a combination of a set of powerful methods such as; actor-critic, policy gradient, 
and Deep Q-Learning. Additionally,  it borrows tricks from other algorithms like Deep Deterministic Policy Gradient (DDPG). TD3 uses actor-critic networks, is 
trained off policy, samples used are from a replay buffer to decollerate them, and  networks are trained with a target network in order to maintain a semi fixed
objective to reach. Additionally, TD3 improves upon DDPG by addressing its main drawbacks: overestimation bias and policy updates, and regularization. 

Neural networks are function approximators, as a result, estimate errors are unavoidable. Additionally, such errors are further exacerbated and accumulated due to
temporal difference updates. TD3 addresses this issue by using two critic networks instead of one. Each of these critic networks provides an Q-value estimate, and 
the minimum of those estimates is used as the target value for both critic networks.

Policy Updates:

The use of target networks improves the stability of the training. However, when it comes to the actor-critic algorithm, there is a further complication that can
potentially lead to divergence. The actor and critic networks are constantly interacting. When the policy update has a large variance (over estimation), 
value estimates will be poor as well. Such poor value estimates will further increases the variance of policy estimates leading to divergent behavior.

TD3 addresses this issue by updating the policy/actor network at a slower rate than that of the critic network. This delay allow the critic network to stabilize and 
thus minimizes the variance before the next policy update.

Target Policy Updates:

Deterministic policy techniques are vulnerable to producing high variance targets due to over-fitting of narrow peaks of value function estimates.

TD3 addresses this issue by adding random noise to the target policy, averaging over mini batches, and clipping the noise to ensure that the resultant values are
within the valid bound of the range of values of the original action.The use of such technique reduces the variance by smoothing/regularizing the value estimate.
