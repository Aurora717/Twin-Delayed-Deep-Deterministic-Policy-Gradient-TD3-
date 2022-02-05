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


TD3 algorithms is implemented using Tensorflow Keras. Where four separate classes were used: Replay-Buffer class, Actor class, Critic class, and an Agent class.

1. Critic Network Architecture:
   The network consists of 2 dense layers with a relu activation 	function, with an output layer that doesn't have an activation function. 
2.  Actor Network Architecture:
    The network consists of 2 dense layers with a relu activation function, an output layer whose size is the amount of output actions with a tanh activation       function to constrict the values within 0-1, since the output is a probability distribution.
    
 The steps for the TD3 algorithm proceed as follows:
 1. Initialize the Replay Buffer with a certain amount of memory.
 2. Initialize the six neural network for TD3: Actor, Target Actor, Critic1, Critic2, Target Critic1, and Target Critic2.
 3. item Randomly sample a batch from the buffer (state $s$, action $a$, next-state s', reward r, done).
    For each item in the batch:
 4. Given a state , the actor network decides the next action a'.
 5. Gaussian noise is then added to that action a', and the result is clamped to a range of valid values. 
 6. The two critic networks are then given the state and the clamped action (s',a'), and each critic network calculates a Qvalue: Q_{1}, Q_{2}.
 7. The minimum of the calculated Qvalues is kept and used in the target. 
 8. Using the target Q-value, we can now calculate the Critic Loss by taking the mean square error of the qvalues and the target.
 9. The critic loss is then back propagated on both critic networks. They are then updated using ADAM optimizer.
 10. Every two iterations, update the actor network via gradient ascent using the output of Critic1 network. Then update the Critic Target networks and the Actor Target networks via Polyak Averaging.

In the implementation of TD3, we apply to a Lunaralander simulator. Lunar Lander is an OpenAI Gym environment where we have a spaceship and two flags placed on the surface. The objective of this environment is to successfully land the spaceship between the two flag.

![0*jY_xUZxuwR2mpGR1](https://user-images.githubusercontent.com/49812606/152655485-475a21f0-c7ce-4c78-80cf-f87212727768.gif)

Results:
![learning_rate](https://user-images.githubusercontent.com/49812606/152655517-cf9d2560-4795-4368-b429-c721c5d79d59.png)
![Noise](https://user-images.githubusercontent.com/49812606/152655521-a42b2fdb-aa80-4753-8162-6d84aff0c5d8.png)
![layer_batch](https://user-images.githubusercontent.com/49812606/152655512-085d1f11-e38f-452d-9cc7-e76f9381f554.png)

