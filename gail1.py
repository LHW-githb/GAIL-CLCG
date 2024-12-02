import numpy as np
import torch
import math
from math import sin, cos, atan2, sqrt
from torch.nn import Module
from scipy.io import loadmat, savemat
from nets import PolicyNetwork, ValueNetwork, Discriminator, SimpleColumnGenerator, ProgNet
from funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch ,get_flat_gradspi,rescale_and_linesearchpi,set_paramspi
import matplotlib.pyplot as plt

from torch import FloatTensor

def normalization(s):
    smax = np.array(
        [2.121320343559642e+04, 4.585058037319488e+02, 0.449400520557010, -0.141897054604164, 0.174532925199433, 21000])
    smin = np.array([3.296435241333737e+02, 2.524581821838549e+02, -1.371547875898444, -1.384000166625609, 0, 20000])
    snorm = 2 * (s - smin) / (smax - smin) - 1
    # snorm = (s - smin) / (smax - smin)
    return snorm


def unnorm(snorm):
    smax = np.array([15.674090553681594])
    smin = np.array([5.269040280106307])
    s = (snorm.detach().numpy() + 1) * (smax - smin) / 2 + smin
    # s = snorm.detach().numpy() * (smax - smin) + smin
    return s

def unnorm_act(snorm):
    smax = np.array([15.674090553681594])
    smin = np.array([5.269040280106307])
    s = (snorm + 1) * (smax - smin) / 2 + smin
    # s = snorm.detach().numpy() * (smax - smin) + smin
    return s
def networkk(input_size,  output_size, num_tasks ,task_id):
    col_gen = SimpleColumnGenerator(input_size, 50, output_size, num_tasks)
    prog_net = ProgNet(col_gen)
    num_tasks = 1
    for _ in range(num_tasks):
        prog_net.addColumn()
    prog_net.freezeAllColumns()
    prog_net.unfreezeColumn(task_id)
    return prog_net

class GAIL1(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()
        self.task_id = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = networkk(self.state_dim, self.action_dim, 1,0)
        self.v = ValueNetwork(self.state_dim)

        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()
        state = FloatTensor(state)
        distb = self.pi.forward(self.task_id, state)

        action = distb.sample().detach().cpu().numpy()

        return action

    def train(self, env, expert, render=False):
        num_iters = 10000
        num_steps_per_iter = 50
        num_steps_per_iter1 = 1
        horizon = None
        lambda_ = 0.001
        gae_gamma = 0.99
        gae_lambda = 0.99
        eps = 0.01
        max_kl = 0.01
        cg_damping = 0.1
        normalize_advantage = True

        opt_d = torch.optim.Adam(self.d.parameters())

        exp_rwd_iter = []

        exp_obs = []
        exp_acts = []
        total_rewards = 0.0
        steps = 0
        while steps < num_steps_per_iter:
            ep_obs = []
            ep_rwds = []

            t = 0
            done = False

            env.modify()

            while not done:
                R = sqrt((env.state[1] - env.xf) ** 2 + (env.state[2] - env.yf) ** 2)
                V = env.state[3]
                Fpa = env.state[4]
                Lamd = atan2((env.yf - env.state[2]), (env.xf - env.state[1]))
                ob = torch.Tensor([R, V, Fpa, Lamd, env.fpa0, env.xf])
                ob = ob.detach().cpu().numpy()
                ob = normalization(ob)
                ob = torch.from_numpy(ob).float()
                act = expert(ob)
                act = unnorm(act)

                ob = ob.numpy()

                if R >= 1:
                    ep_obs.append(ob)
                    exp_obs.append(ob)
                    exp_acts.append(act)

                ob, done = env.step(act)


                t += 1

                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break

            steps += 1
            print(
                "Iterations_collection: {}"
                .format(steps)
            )

        exp_obs = FloatTensor(np.array(exp_obs))
        exp_acts = FloatTensor(np.array(exp_acts))

        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < num_steps_per_iter1:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False

                env.modify()
                pltx = []
                plty = []
                pltt = []
                pltac = []
                while not done:
                    R = sqrt((env.state[1] - env.xf) ** 2 + (env.state[2] - env.yf) ** 2)
                    V = env.state[3]
                    Fpa = env.state[4]
                    Lamd = atan2((env.yf - env.state[2]), (env.xf - env.state[1]))
                    ob = torch.Tensor([R, V, Fpa, Lamd, env.fpa0, env.xf])
                    ob = ob.detach().cpu().numpy()
                    ob = normalization(ob)
                    ob = torch.from_numpy(ob).float()
                    ob = ob.numpy()
                    act = self.act(ob)
                    act = unnorm_act(act)
                    ep_obs.append(ob)
                    obs.append(ob)

                    ep_acts.append(act)
                    acts.append(act)

                    ob, done = env.step(act)

                    ep_gms.append(gae_gamma ** t)
                    ep_lmbs.append(gae_lambda ** t)

                    t += 1

                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break
                steps += 1

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)) \
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs
                total_rewards += ep_costs.sum().item()
                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[i:]) for i in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                ep_deltas = ep_costs.unsqueeze(-1) \
                            + gae_gamma * next_vals \
                            - curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            print(
                "Iterations: {},   err_angle: {}   err_x: {} err_y: {}  reward: {}"
                .format(i + 1, -80 - env.state[4] / (math.pi / 180),  env.xf - env.state[1],
                        env.yf - env.state[2],total_rewards/t)
            )

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)
            exp_scores_sum = exp_scores.sum().item()
            nov_scores_sum = nov_scores.sum().item()

            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                   + torch.nn.functional.binary_cross_entropy_with_logits(
                nov_scores, torch.ones_like(nov_scores)
            )
            loss.backward()

            opt_d.step()

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v) \
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi.columns[-1]).detach()
            old_distb = self.pi.forward(self.task_id, obs)

            def L():
                distb = self.pi.forward(self.task_id, obs)

                return (advs * torch.exp(
                    distb.log_prob(acts)
                    - old_distb.log_prob(acts).detach()
                )).mean()

            def kld():
                distb = self.pi.forward(self.task_id, obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p))) \
                        .sum(-1) \
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                    ).mean()

            grad_kld_old_param = get_flat_gradspi(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_gradspi(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_gradspi(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearchpi(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = ((-1) * gms * self.pi.forward(self.task_id, obs).log_prob(acts)) \
                .mean()
            grad_disc_causal_entropy = get_flat_gradspi(
                disc_causal_entropy, self.pi
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_paramspi(self.pi, new_params)

            if hasattr(self, "pi"):
                torch.save(
                    self.pi.state_dict(), f"task1_[{i}].ckpt"
                )

        return exp_rwd_mean, rwd_iter_means