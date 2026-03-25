from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


from .strategy_features import StrategyFeatureBuilder
from .types import Row


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ModuleNotFoundError as exc:
        raise ValueError("DQN requires PyTorch. Install it first (e.g. `pip install torch`).") from exc

    return torch, nn, optim


class DQNPolicyNetwork:
    def __init__(self, state_size: int, action_size: int) -> None:
        torch, nn, _ = _torch()
        self.module = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )
        self.torch = torch

    def to(self, device):
        self.module = self.module.to(device)
        return self


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        from collections import deque

        self.memory = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(args)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class TradingEnv:
    def __init__(
        self,
        rows: Sequence[Row],
        features: StrategyFeatureBuilder,
        means: Sequence[float],
        stds: Sequence[float],
        lot_size: float = 0.001,
        start_balance: float = 1000.0,
        window: int = 6000,
        fee: float = 0.004,
    ) -> None:
        self.rows = list(rows[-window:])
        self.features = features
        self.means = list(means)
        self.stds = list(stds)
        self.lot_size = lot_size
        self.start_balance = start_balance
        self.fee = fee
        self.reset()

    def reset(self) -> List[float]:
        self.t = 0
        self.balance = self.start_balance
        self.holding_num = 0.0
        self.done = False
        return self._get_state()

    def _feature_vector(self, row: Row) -> List[float]:
        values = self.features.transform([row])[0]
        return [(values[i] - self.means[i]) / self.stds[i] for i in range(len(values))]

    def _get_state(self) -> List[float]:
        row = self.rows[self.t]
        feature_values = self._feature_vector(row)
        return [
            *feature_values,
            self.holding_num / max(1e-12, self.lot_size),
            self.balance / max(1e-12, self.start_balance),
        ]

    def step(self, action: int) -> Tuple[List[float], float, bool]:
        row = self.rows[self.t]
        price = float(row.get("close", 0.0))
        old_value = self.balance + (self.holding_num * price)
        reward = 0.0
        if action == 1:
            if self.holding_num == 0.0:
                cost = price * self.lot_size
                total_cost = cost * (1.0 + self.fee)
                if self.balance >= total_cost:
                    self.balance -= total_cost
                    self.holding_num = self.lot_size
                else:
                    reward = -0.0002
            else:
                reward = -0.0002
        elif action == 2:
            if self.holding_num > 0.0:
                revenue = price * self.lot_size
                net_revenue = revenue * (1.0 - self.fee)
                self.balance += net_revenue
                self.holding_num = 0.0
            else:
                reward = -0.0002

        self.t += 1
        self.done = self.t >= len(self.rows) - 1
        if self.done:
            return self._get_state(), reward, self.done
        new_price = float(self.rows[self.t].get("close", 0.0))
        new_value = self.balance + (self.holding_num * new_price)
        safe_old = old_value if old_value > 1e-12 else 1e-12
        safe_new = new_value if new_value > 1e-12 else 1e-12
        reward += math.log(safe_new / safe_old)
        return self._get_state(), reward, self.done

    def current_profit(self) -> float:
        if not self.rows:
            return 0.0
        price = float(self.rows[min(self.t, len(self.rows) - 1)].get("close", 0.0))
        portfolio_value = self.balance + (self.holding_num * price)
        return portfolio_value - self.start_balance


def serialize_state_dict(state_dict: Dict[str, object]) -> Dict[str, object]:
    serialized: Dict[str, object] = {}
    for key, tensor in state_dict.items():
        serialized[key] = tensor.detach().cpu().tolist()
    return serialized


def load_state_dict_from_json(model, state_dict_json: Dict[str, object]) -> None:
    torch, _, _ = _torch()
    state_dict = model.state_dict()
    converted = {}
    for key, value in state_dict_json.items():
        if key in state_dict:
            converted[key] = torch.tensor(value, dtype=state_dict[key].dtype)
    state_dict.update(converted)
    model.load_state_dict(state_dict)


def dqn_q_values(bundle: Dict[str, object], state_vector: Sequence[float]) -> List[float]:
    if "dqn_state_dict" not in bundle:
        return [0.0, 0.0, 0.0]
    torch, _, _ = _torch()
    state_size = int(bundle.get("dqn_state_size", len(state_vector)))
    action_size = int(bundle.get("dqn_action_size", 3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQNPolicyNetwork(state_size=state_size, action_size=action_size).to(device)
    load_state_dict_from_json(net.module, bundle["dqn_state_dict"])
    net.module.eval()
    with torch.no_grad():
        state_t = torch.tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)
        return net.module(state_t).squeeze(0).detach().cpu().tolist()


def train_dqn_policy(
    train_rows: Sequence[Row],
    features: StrategyFeatureBuilder,
    means: Sequence[float],
    stds: Sequence[float],
    *,
    episodes: int = 120,
    gamma: float = 0.95,
    learning_rate: float = 1e-3,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
) -> Tuple[Dict[str, object], List[float], List[float], List[float]]:
    torch, nn, optim = _torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DQN] Training on device: {device}")
    env = TradingEnv(train_rows, features=features, means=means, stds=stds)
    state_size = len(env.reset())
    action_size = 3
    policy = DQNPolicyNetwork(state_size, action_size).to(device).module
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    memory = ReplayBuffer(20000)
    batch_size = 64
    epsilon = 1.0
    episode_rewards: List[float] = []
    action_returns = [0.0, 0.0, 0.0]
    action_counts = [0.0, 0.0, 0.0]

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        while not env.done:
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(torch.argmax(policy(state_t)).item())
            next_state, reward, done = env.step(action)
            action_returns[action] += reward
            action_counts[action] += 1.0
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                b_states = torch.tensor([t[0] for t in transitions], dtype=torch.float32, device=device)
                b_actions = torch.tensor([t[1] for t in transitions], dtype=torch.long, device=device)
                b_rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32, device=device)
                b_next_states = torch.tensor([t[3] for t in transitions], dtype=torch.float32, device=device)
                b_dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32, device=device)
                current_q = policy(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = policy(b_next_states).max(1)[0]
                    target_q = b_rewards + (gamma * max_next_q * (1 - b_dones))
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        episode_rewards.append(total_reward)
        episode_profit = env.current_profit()
        print(
            f"[DQN] Episode {episode + 1}/{episodes} | reward={total_reward:.6f} | profit={episode_profit:.6f}"
        )
        if epsilon > epsilon_min:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    avg_action_returns = [action_returns[idx] / action_counts[idx] if action_counts[idx] else 0.0 for idx in range(action_size)]
    return (
        serialize_state_dict(policy.state_dict()),
        avg_action_returns,
        [float(epsilon)],
        episode_rewards,
    )
