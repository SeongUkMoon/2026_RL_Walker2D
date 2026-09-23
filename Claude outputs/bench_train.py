"""Time-budgeted CPU PPO benchmark. Usage: python bench_train.py <tag> <xml or 'stock'> <minutes> [reward_mode]"""
import os, sys, time, json
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

torch.set_num_threads(2)
tag, xml, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
reward_mode = sys.argv[4] if len(sys.argv) > 4 else "default"
budget = minutes * 60


class NLReward(gym.Wrapper):
    """Stand-in for an AI-generated reward: 'move forward, keep the torso high and level, don't waste energy'."""
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        d = self.env.unwrapped.data
        x_vel = info["x_velocity"]
        z = d.qpos[1]
        angle = d.qpos[2]
        r = 1.0 * x_vel + 1.0 * (z > 0.45) - 0.5 * abs(angle) - 1e-3 * float(np.square(action).sum())
        return obs, r, term, trunc, info


def make_env():
    kw = dict()
    if xml != "stock":
        kw = dict(xml_file=os.path.abspath(xml), healthy_z_range=(0.35, 3.0), healthy_angle_range=(-1.2, 1.2))
    e = gym.make("Walker2d-v5", **kw)
    if reward_mode == "nl":
        e = NLReward(e)
    return e


class TimeLog(BaseCallback):
    def __init__(self):
        super().__init__(); self.t0 = time.time(); self.rows = []; self.next = 30
    def _on_step(self):
        el = time.time() - self.t0
        if el >= self.next:
            self.next += 30
            ep = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
            ln = [e["l"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
            row = dict(t=round(el), steps=self.num_timesteps, fps=round(self.num_timesteps / el),
                       rew=round(float(np.mean(ep)), 1) if ep else None,
                       ep_len=round(float(np.mean(ln))) if ln else None)
            self.rows.append(row); print(json.dumps(row), flush=True)
        return el < budget


venv = make_vec_env(make_env, n_envs=8, seed=0)
model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, n_epochs=10, learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, verbose=0, device="cpu")
cb = TimeLog()
model.learn(total_timesteps=int(1e9), callback=cb)
model.save(f"ppo_{tag}")
json.dump(cb.rows, open(f"bench_{tag}.json", "w"))

# short rollout of the final policy -> frame strip
os.environ.setdefault("MUJOCO_GL", "egl")
e = make_env(); e = gym.make("Walker2d-v5", render_mode="rgb_array", **({} if xml == "stock" else dict(
    xml_file=os.path.abspath(xml), healthy_z_range=(0.35, 3.0), healthy_angle_range=(-1.2, 1.2))))
obs, _ = e.reset(seed=1); frames = []; ret = 0; x0 = e.unwrapped.data.qpos[0]
for t in range(400):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = e.step(a); ret += r
    if t % 50 == 0: frames.append(e.render())
    if term or trunc: break
x1 = e.unwrapped.data.qpos[0]
print(json.dumps(dict(final_eval_steps=t + 1, final_eval_return=round(float(ret), 1), distance_m=round(float(x1 - x0), 2))))
import imageio.v2 as imageio
imageio.imwrite(f"final_{tag}.png", np.concatenate(frames[:8], axis=1))
