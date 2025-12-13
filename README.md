# Gymjs

Gymjs is an open source JS library for developing environments for reinforcement learning by providing a standard API, similar to Python's gym, and a couple of compliant environments like Cartpole and Pendulum.

## Installation

```bash
npm install gymjs
```

## API

Gymjs's API is very similar to that of gymnasium. Python code for running CartPole's environment:

```py
import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

Equivalent gymjs code:

```ts
import { CartPoleEnv } from 'gymjs/classic_control';
const env = new CartPoleEnv();

let [observation, info] = env.reset();
for (let i = 0; i < 1000; i++) {
  let action = env.actionSpace.sample();
  let [observation, reward, terminated, truncated, info] =
    await env.step(action);

  if (terminated || truncated) {
    let [observation, info] = env.reset();
  }
}
env.close();
```
