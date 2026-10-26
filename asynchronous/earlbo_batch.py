"""
EARL-BO, synchronous-batch form: q query points per Bayesian-optimisation iteration.

The sequential method (see earlbo_seq.py in the sequential package) widened to q policy heads: the actor outputs q*d
means (head j = entries j*d .. (j+1)*d), the critic q values, and PPO's ratio, clipping and advantage are per head.
One virtual step evaluates the q points at once: the q outcomes are one draw from the joint GP posterior of the q
points, the team reward is the improvement of their maximum over the running incumbent, the q rows are appended and the
GP conditioned once. The training targets of both updates are the discounted team returns, z-scored over the [T, q]
buffer and shared by all heads. The teacher is batch TuRBO (q Thompson samples) rolled out on the same virtual data.
Schedule and everything else as in the sequential form: 4000 episodes of `horizon` steps, an update every 50 episodes,
400 imitation episodes first (10 Adam epochs per update, two layers then frozen), PPO afterwards (10 epochs per update);
the deployed batch is the q mean actions. Fallback: when 15 consecutive updates (or the last update) show a mean
virtual reward below 1e-5, training stops and one TuRBO batch on the real data is deployed instead.

Coordinate conventions: the runner maps the real box [lb, ub]^d to N = [-1, 1]^d once (Domain) and standardises y once
per iteration on the real observations (Scaler); everything inside RL_BO.evaluate works in these units.
"""
import time
import traceback
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from torch.distributions import MultivariateNormal

from turbo import Turbo1
from utils import from_unit_cube, to_unit_cube

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Predicted variances smaller than 0. Setting those variances to 0.",
                        module="sklearn.gaussian_process._gpr")

device = torch.device("cpu")


# ------------------------------------------------------------------ benchmark functions (maximised: real(x) = -f(x))
class Ackley:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        term1 = -0.2 * np.sqrt(np.sum(x ** 2, axis=1) / self.dimension)
        term2 = np.sum(np.cos(2 * np.pi * x), axis=1) / self.dimension
        return -(-20 * np.exp(term1) - np.exp(term2) + 20 + np.e)


class Sum_square:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        coefficients = np.arange(1, self.dimension + 1)
        return -np.sum(coefficients * x ** 2, axis=1)


class Levy:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        w = 1 + (x - 1) / 4
        term1 = (np.sin(np.pi * w[:, 0])) ** 2
        wi = w[:, :-1]
        term_middle = np.sum((wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2), axis=1)
        term_last = (w[:, -1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[:, -1])) ** 2)
        return -(term1 + term_middle + term_last)


class Rosenbrock:
    def __init__(self, dimension):
        self.dimension = dimension

    def real(self, x):
        sum_terms = 100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2
        return -(np.sum(sum_terms, axis=1))


TEST_FUNCTIONS = {'Ackley': Ackley, 'Levy': Levy, 'Rosenbrock': Rosenbrock, 'Sum_square': Sum_square}


# ------------------------------------------------------------------ coordinate handling
class Scaler:
    """Standardisation of y (population std), fitted once per BO iteration on the real observations."""

    def __init__(self):
        self.mu = None
        self.std = None

    def fit(self, x):
        self.mu = np.mean(x, 0)
        self.std = np.std(x, 0)
        self.std = np.where(self.std < 1e-12, 1.0, self.std)   # a constant column would divide by zero

    def transform(self, x):
        return (x - self.mu) / self.std

    def inverse_transform_mean(self, x):
        return x * self.std + self.mu


class Domain:
    """Fixed affine map between the real box [lb, ub]^d and N = [-1, 1]^d (the only transform of x)."""

    def __init__(self, lb, ub):
        self.lb = np.asarray(lb, dtype=float).reshape(-1)
        self.ub = np.asarray(ub, dtype=float).reshape(-1)
        assert self.lb.shape == self.ub.shape and np.all(self.ub > self.lb), "domain bounds must satisfy ub > lb"
        self.c = (self.ub + self.lb) / 2.0
        self.hw = (self.ub - self.lb) / 2.0

    def to_norm(self, x_real, tol=1e-9):
        x = (np.asarray(x_real, dtype=float) - self.c) / self.hw
        assert x.min() >= -1.0 - tol and x.max() <= 1.0 + tol, \
            f"data outside the domain: min={x.min()}, max={x.max()} (N units)"
        return np.clip(x, -1.0, 1.0)   # removes round-off only (see the assert)

    def to_real(self, x_norm):
        return self.c + self.hw * np.asarray(x_norm, dtype=float)


def conditioning_gp(fitted_gpr):
    """Fixed-hyperparameter copy of a fitted sklearn GP: its fit() only conditions on the data it is given.
    sklearn starts every fit from a clone of `kernel`, so the copy's kernel must be the fitted kernel_."""
    return GaussianProcessRegressor(kernel=deepcopy(fitted_gpr.kernel_), optimizer=None,
                                    normalize_y=False, alpha=fitted_gpr.alpha)


# ------------------------------------------------------------------ replay memory and PPO agent
class Memory:
    """One update window: actions (q points each), their per-head log-probabilities, team rewards, episode ends, and
    for every stored step the data set [X_N | y_s] its state was encoded from (re-encoded with gradient at the update)."""

    def __init__(self):
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.snapshots = []

    def clear_memory(self):
        self.__init__()


class ActorCritic(nn.Module):
    """q heads: the actor outputs q*d means (head j = entries j*d .. (j+1)*d), the critic q values."""

    def __init__(self, num_state, num_action, action_std, num_batch_actions):
        super(ActorCritic, self).__init__()
        self.num_action = num_action
        self.num_batch_actions = num_batch_actions
        self.actor = nn.Sequential(
            nn.Linear(num_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_action * num_batch_actions),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(num_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_batch_actions),
        )
        self.action_var = torch.full((num_batch_actions, num_action), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # Sample q points from the current policy and store them with their per-head log-probabilities
        action_mean = self.actor(state).reshape(self.num_batch_actions, self.num_action)
        cov_mat = torch.diag_embed(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()                       # [q, d]
        action = action * (~torch.isnan(action))
        action = torch.clamp(action, -1., 1.)
        action_logprob = dist.log_prob(action)       # [q]
        memory.actions.append(action.cpu())
        memory.logprobs.append(action_logprob.cpu())
        return action.detach()

    def act_mean(self, state):
        # Mean actions: [q, d] for one state, [T, q, d] for a batch of states
        action_mean = self.actor(state)
        action_mean = action_mean * (~torch.isnan(action_mean))
        return action_mean.reshape(action_mean.shape[:-1] + (self.num_batch_actions, self.num_action))

    def evaluate(self, state, action):
        # state [T, 16], action [T, q, d] -> log-probabilities, values and entropies, all [T, q]
        action_mean = self.actor(state).reshape(-1, self.num_batch_actions, self.num_action)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, state_value, dist_entropy


class PPO_Agent:
    def __init__(self, num_state, num_action, num_batch_actions):
        self.num_state = num_state
        self.num_action = num_action
        self.num_batch_actions = num_batch_actions
        self.learning_rate = 0.001
        self.action_std = 0.1
        self.K_epochs = 10            # PPO epochs per update
        self.imitation_epochs = 10    # Adam epochs per imitation update
        self.eps_clip = 0.2
        self.betas = (0.9, 0.999)
        self.action_decay = 0.99
        self.gamma = 0.95
        self.VF_coeff = 0.5
        self.Entropy_coeff = 0.01
        self.action_std_min = 0.01
        self.freeze_num = 2           # layers of actor and critic frozen by transfer_learning

        self.policy = ActorCritic(self.num_state, self.num_action, self.action_std, self.num_batch_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, betas=self.betas)
        self.policy_old = ActorCritic(self.num_state, self.num_action, self.action_std, self.num_batch_actions).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def get_action(self, state, memory):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = state.to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy()

    def get_action_eval(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = state.to(device)
        return self.policy_old.act_mean(state).cpu().data.numpy().reshape(self.num_batch_actions, self.num_action)

    def _targets(self, memory):
        # Training targets [T, q] of one buffer, used by both updates: the team reward of every step is discounted
        # (reset at episode ends), copied to the q heads and z-scored over the [T, q] array, with
        # s^2 = sum (G - m)^2 / (q (T - 1)) so that q = 1 is exactly the sequential normalisation.
        T, q = len(memory.rewards), self.num_batch_actions
        assert T >= 2, "the normalisation needs at least two stored steps"
        r = np.asarray(memory.rewards, dtype=float)
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            if memory.is_terminals[t]:
                running = 0.0
            running = r[t] + (self.gamma * running)
            G[t] = running
        G = torch.FloatTensor(np.repeat(G[:, None], q, axis=1)).to(device)
        m = G.mean()
        s = G.std() * float(np.sqrt((T * q - 1.0) / (q * (T - 1.0))))
        return (G - m) / (s + 1e-5)

    def _sync_old_policy(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy.action_var = torch.full((self.num_batch_actions, self.num_action),
                                            self.action_std * self.action_std).to(device)
        self.policy_old.action_var = torch.full((self.num_batch_actions, self.num_action),
                                                self.action_std * self.action_std).to(device)

    def update(self, memory, encoder, encoder_optimizer):
        """One PPO update, every term per head [T, q]; the stored data sets are re-encoded with gradient."""
        self.action_std = max(self.action_std * self.action_decay, self.action_std_min)
        rewards = self._targets(memory)
        assert len(memory.snapshots) == len(memory.actions), "one data-set snapshot per stored transition"
        x_pad, mask = pad_snapshots(memory.snapshots)
        old_actions = torch.stack(memory.actions).to(device).detach()     # [T, q, d]
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()   # [T, q]
        for _ in range(self.K_epochs):
            states = encoder.forward_batch(x_pad, mask)
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.VF_coeff * self.MseLoss(state_values,
                                                                           rewards) - self.Entropy_coeff * dist_entropy
            self.optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            encoder_optimizer.step()
        self._sync_old_policy()

    def update_initial(self, memory, encoder, encoder_optimizer):
        """One imitation update: the mean actions regress the teacher's q points, the critic the targets."""
        rewards = self._targets(memory)
        assert len(memory.snapshots) == len(memory.actions), "one data-set snapshot per stored transition"
        x_pad, mask = pad_snapshots(memory.snapshots)
        old_actions = torch.stack(memory.actions).to(device).detach()   # [T, q, d]
        for _ in range(self.imitation_epochs):
            states = encoder.forward_batch(x_pad, mask)
            _, state_values, _ = self.policy.evaluate(states, old_actions)
            actions = self.policy.act_mean(states)
            loss = self.MseLoss(state_values, rewards) + self.MseLoss(actions, old_actions)
            self.optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            encoder_optimizer.step()
        self._sync_old_policy()

    def transfer_learning(self):
        """Rebuild the policy with the first `freeze_num` layers of actor and critic frozen and a new Adam over the rest."""
        initial_policy_params = deepcopy(self.policy.state_dict())
        self.policy = ActorCritic(self.num_state, self.num_action, self.action_std, self.num_batch_actions).to(device)
        self.policy.load_state_dict(initial_policy_params)
        for w in range(self.freeze_num):
            self.policy.actor[2 * w].weight.requires_grad = False
            self.policy.actor[2 * w].bias.requires_grad = False
            self.policy.critic[2 * w].weight.requires_grad = False
            self.policy.critic[2 * w].bias.requires_grad = False
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=self.learning_rate,
            betas=self.betas
        )
        self.policy_old = ActorCritic(self.num_state, self.num_action, self.action_std, self.num_batch_actions).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())


# ------------------------------------------------------------------ TuRBO teacher and virtual environment
class ModifiedTurbo1(Turbo1):
    """Batch TuRBO-1 used as a q-point proposal generator on whatever data set it is handed (trust-region length 0.8)."""

    def __init__(self, X, Y, lb, ub, num_batch_actions):
        n_init = len(X)
        super().__init__(f=lambda x: 0,
                         lb=lb,
                         ub=ub,
                         n_init=n_init,
                         max_cholesky_size=2000,
                         max_evals=max(n_init + 10, 100, num_batch_actions + 1),
                         n_training_steps=50,
                         batch_size=num_batch_actions,   # q Thompson samples -> q proposals
                         device="cpu",
                         dtype="float64")
        self.set_data(X, Y)

    def set_data(self, X, y):
        # The data set the next proposal is based on (N inputs, y_s values; TuRBO minimises, hence -y). Copies.
        X = np.array(X, dtype=float, copy=True)
        y = np.array(y, dtype=float, copy=True).reshape(-1, 1)
        assert len(X) == len(y), "teacher data: X and y must have the same number of rows"
        self._X = X
        self._fX = -y
        self.n_evals = len(self._X)

    def optimize(self):
        X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)
        fX = deepcopy(self._fX).ravel()
        X_cand, y_cand, _ = self._create_candidates(
            X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
        )
        X_next = self._select_candidates(X_cand, y_cand)
        X_next = from_unit_cube(X_next, self.lb, self.ub)
        return X_next   # (q, d)


class Env_encoder:
    """Virtual BO environment in N and y_s: state = encoded data set, action = q points, reward = team improvement."""

    def __init__(self, model, encoder, X_train_org, y_train_org, num_batch_actions):
        self.num_state = 16
        self.num_action = X_train_org.shape[1]
        self.num_batch_actions = num_batch_actions
        self.lb = -np.ones(self.num_action)
        self.ub = np.ones(self.num_action)
        self.encoder = encoder
        self.model = model   # fixed-hyperparameter GP (conditioning_gp): fit() here only conditions on data
        self.X_train_org = X_train_org
        self.y_train_org = y_train_org
        self.turbo = ModifiedTurbo1(X_train_org, y_train_org, self.lb, self.ub, num_batch_actions)

    def snapshot(self):
        # The encoder's input for the current state, float32 [X_train | y_train]; stored in the replay memory
        return np.concatenate([self.X_train, self.y_train], axis=1).astype(np.float32)

    def _encode(self):
        X_pt = torch.tensor(self.X_train, dtype=torch.float32).to(device)
        y_pt = torch.tensor(self.y_train, dtype=torch.float32).to(device)
        return self.encoder(torch.cat([X_pt, y_pt], dim=1))

    def reset(self):
        self.X_train = self.X_train_org
        self.y_train = self.y_train_org
        self.model.fit(self.X_train, self.y_train)
        self.state = self._encode()
        return self.state.to(device)

    def step(self, action_):
        q = self.num_batch_actions
        action = np.asarray(action_, dtype=float).reshape(q, -1)
        assert action.shape == (q, self.num_action), f"step expects q x d = {(q, self.num_action)} values"
        self.y_max = np.max(self.y_train)
        # One draw from the joint posterior of the q points (the covariance carries the white-noise term)
        mean, cov = self.model.predict(action, return_cov=True)
        mean = np.asarray(mean).reshape(-1, )
        cov = np.asarray(cov).reshape(q, q)
        next_observation = np.random.multivariate_normal(mean, cov + 1e-9 * np.eye(q)).reshape(-1, )
        reward = max(0, np.max(next_observation) - self.y_max)
        self.X_train = np.vstack((self.X_train, action))
        self.y_train = np.vstack((self.y_train, next_observation.reshape(-1, 1)))
        self.model.fit(self.X_train, self.y_train)
        self.state = self._encode()
        return self.state.to(device), reward

    def turbo_acquisition(self, X=None, y=None):
        """TuRBO proposal (q, d) in N for the data set (X, y); with no arguments, for the real data (the fallbacks)."""
        assert (X is None) == (y is None), "pass both X and y, or neither"
        if X is None:
            X, y = self.X_train_org, self.y_train_org
        self.turbo.set_data(X, y)
        action = np.array(self.turbo.optimize(), dtype=float).reshape(self.num_batch_actions, -1)
        assert action.min() >= -1.0 - 1e-9 and action.max() <= 1.0 + 1e-9, "teacher proposal outside [-1, 1]"
        return np.clip(action, -1.0, 1.0)


# ------------------------------------------------------------------ attention-based DeepSets encoder
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(K.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output


class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSetEncoder, self).__init__()
        self.attention = AttentionLayer(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = x.mean(dim=0)
        x = self.fc2(x)
        return x

    def forward_batch(self, x, mask):
        # forward() for a zero-padded batch of sets, x [B, N, D], mask [B, N] (True = real row): padded keys leave
        # the softmax, padded rows leave the mean
        Q, K, V = self.attention.query(x), self.attention.key(x), self.attention.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
        scores = scores.masked_fill(~mask[:, None, :], float('-inf'))
        out = torch.matmul(F.softmax(scores, dim=-1), V)
        out = F.relu(self.fc1(out)) * mask[..., None]
        pooled = out.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.fc2(pooled)


def pad_snapshots(snapshots):
    """List of float32 arrays (n_i, D) -> zero-padded x [B, Nmax, D] and mask [B, Nmax] (True = real row)."""
    B, D = len(snapshots), snapshots[0].shape[1]
    Nmax = max(s.shape[0] for s in snapshots)
    x = torch.zeros(B, Nmax, D, dtype=torch.float32)
    mask = torch.zeros(B, Nmax, dtype=torch.bool)
    for i, s in enumerate(snapshots):
        x[i, :s.shape[0]] = torch.as_tensor(s, dtype=torch.float32)
        mask[i, :s.shape[0]] = True
    return x.to(device), mask.to(device)


# ------------------------------------------------------------------ the acquisition: train a policy, deploy its mean actions
class RL_BO:
    def __init__(self):
        self.horizon = 5
        self.max_episodes = 4000
        self.update_episode = 50
        self.off_policy_episodes = 400
        self.num_batch_actions = None   # q; set before evaluate

    def evaluate(self, model, X_train_org, y_train_org):
        """Train the agent on the virtual environment built from `model` (a fitted GP) and the data (X in N, y in y_s);
        return the next batch (q, d) in N."""
        torch.manual_seed(1)
        np.random.seed(1)
        q = self.num_batch_actions
        assert isinstance(q, int) and q >= 1, f"num_batch_actions must be a positive int before evaluate: {q}"
        encoder = DeepSetEncoder(input_dim=X_train_org.shape[1] + 1, hidden_dim=64, output_dim=16).to(device)
        env = Env_encoder(conditioning_gp(model), encoder, X_train_org, y_train_org, q)
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001, betas=(0.9, 0.999))
        memory = Memory()
        agent = PPO_Agent(env.num_state, env.num_action, q)

        return_sum = 0
        final_average_score = 0
        no_improvement_count = 0
        no_improvement_threshold = 15

        for e in range(1, self.max_episodes + 1):
            state = env.reset().to(device)
            total_reward = 0
            for k in range(self.horizon):
                memory.snapshots.append(env.snapshot())
                if e <= self.off_policy_episodes:
                    # Teacher step: batch TuRBO proposes on the current (real + virtual) data; store the log-probabilities
                    turbo_action = env.turbo_acquisition(env.X_train, env.y_train)   # (q, d)
                    action_tensor = torch.FloatTensor(turbo_action).to(device)
                    action_mean = agent.policy.actor(state).reshape(q, env.num_action)
                    cov_mat = torch.diag_embed(agent.policy.action_var).to(device)
                    dist = MultivariateNormal(action_mean, cov_mat)
                    memory.actions.append(action_tensor.cpu())
                    memory.logprobs.append(dist.log_prob(action_tensor).cpu())
                    action = turbo_action
                else:
                    action = agent.get_action(state, memory)
                next_state, reward = env.step(action)
                total_reward += reward
                memory.rewards.append(reward)
                memory.is_terminals.append(k == self.horizon - 1)
                state = next_state.detach().clone()
            return_sum += total_reward

            if e % self.update_episode == 0:
                if e <= self.off_policy_episodes:
                    agent.update_initial(memory, encoder, encoder_optimizer)
                    agent.transfer_learning()
                else:
                    agent.update(memory, encoder, encoder_optimizer)
                memory.clear_memory()
                score = return_sum / self.update_episode
                print(f'Episode {e}, Average score: {score}')
                return_sum = 0

                if score < 1e-5:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                # TuRBO fallback: 15 consecutive updates without reward end the training
                if no_improvement_count >= no_improvement_threshold:
                    return env.turbo_acquisition().reshape(q, -1)
                if e == self.max_episodes:
                    final_average_score = score

        if final_average_score < 1e-5:
            x_next = env.turbo_acquisition()
        else:
            state = env.reset()
            x_next = agent.get_action_eval(state)
        return x_next.reshape(q, -1)


def run_single_experiment(seed, X_init, func_name, dimension, num_iterations, domain_lb, domain_ub, horizon,
                          num_batch_actions):
    """One batch EARL-BO run (q = num_batch_actions points per iteration) from the initial design X_init (real units) on
    `func_name` in [domain_lb, domain_ub]^dimension. Returns (regrets, decision_times): regrets has num_iterations + 1
    entries (index 0 = the initial design), in the minimisation units of the benchmark (optimum 0); decision_times has
    one entry per iteration (seconds). None on error."""
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        test_func = TEST_FUNCTIONS[func_name](dimension)
        y_max_real = 0

        X_train = X_init
        y_train = test_func.real(X_train).reshape(-1, 1)
        regrets = [y_max_real - np.max(y_train)]

        domain = Domain([domain_lb] * dimension, [domain_ub] * dimension)
        kernel = RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)
        scaler_y = Scaler()
        acq_func = RL_BO()
        acq_func.horizon = horizon
        acq_func.num_batch_actions = num_batch_actions
        decision_times = []

        for j in range(num_iterations):
            scaler_y.fit(y_train)
            y_train_scaled = scaler_y.transform(y_train)
            X_train_norm = domain.to_norm(X_train)
            gpr.fit(X_train_norm, y_train_scaled)

            start_time = time.time()
            print(f"\n===== Iteration {j + 1}/{num_iterations}, seed {seed} =====")
            X_next = acq_func.evaluate(gpr, X_train_norm, y_train_scaled)
            decision_times.append(time.time() - start_time)
            print(f"Decision {j + 1}/{num_iterations} took {decision_times[-1]:.1f} s")

            X_next = domain.to_real(X_next)
            y_next = test_func.real(X_next).reshape(-1, 1)
            X_train = np.vstack((X_train, X_next))
            y_train = np.vstack((y_train, y_next))
            regrets.append(y_max_real - np.max(y_train))

        return regrets, decision_times

    except Exception:
        print(f"Error in experiment with seed {seed}:")
        traceback.print_exc()
        return None
