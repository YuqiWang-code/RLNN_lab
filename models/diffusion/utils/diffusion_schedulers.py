"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math

import numpy as np
import torch
import torch.nn.functional as F


class CategoricalDiffusion(object):
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T

        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
            alpha = 1.0 - self.beta
            # ᾱ(0)=1, ᾱ(t)=∏_{s=1..t} alpha[s]
            self.alphabar = np.concatenate([[1.0], np.cumprod(alpha)])
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
                0)  # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))

        ones = np.array([[1, 0], [1, 0]], dtype=float).reshape((1, 2, 2))

        self.Qs = (1 - beta) * eye + beta * ones

        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def sample(self, x0_onehot, t, u_shared=None):
        device = x0_onehot.device
        x0_val = x0_onehot[..., 1]

        if isinstance(t, torch.Tensor):
            t_idx = t.detach().cpu().numpy()
        else:
            t_idx = np.asarray(t)
        alpha_bar = torch.from_numpy(self.alphabar[t_idx]).to(device).float()  # [B]
        B = x0_val.shape[0]
        shape_each = x0_val.shape[1:]

        if u_shared is None:
            U = torch.rand((B,) + shape_each, device=device)
        else:
            U = u_shared
        k = (U < alpha_bar.view(B, *([1] * len(shape_each)))).long()

        # 核心：未知位置用 0.5、已知位置用真值
        # xt ∈ {0, 0.5, 1} ：未知=0.5，已知0=0，已知1=1
        xt = k * x0_val.float() + (1.0 - k) * 0.0

        return xt, U

    def consistency_sample(self, x0_onehot, t, t2):
        x_t, U = self.sample(x0_onehot, t, u_shared=None)
        x_t2, _ = self.sample(x0_onehot, t2, u_shared=U)
        return x_t, x_t2

class InferenceSchedule(object):
    def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i):
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "abs":
            t1 = self.T - int(
                1 / (self.inference_T - float(i) + 1) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int(
                1 / (self.inference_T - float(i + 1) + 1) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        else:
            raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
