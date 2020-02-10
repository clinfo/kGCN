import math
import copy
import itertools
from operator import mul

import numpy as np
import scipy.optimize
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchex.nn as exnn

from torchvision import datasets, transforms


def normalize_1d(x, dim=0):
    mean = x.mean(dim)
    std = x.std(dim) + 1e-25
    norm_x = (x - mean) / std
    return norm_x, mean, std

def normalize_col(x):
    _x = copy.deepcopy(x)
    dim = 0
    row, col = x.shape
    mean = x.mean(dim)
    std = x.std(dim) + 1e-25
    for i in range(row):
        x[i, :] -= mean
        x[i, :] /= std
    return x, mean, std


class SimpleNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleNetwork, self).__init__()
        h1_dim = 16
        h2_dim = 32
        h3_dim = 64
        self.net1 = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.Tanh(),
            nn.Linear(h1_dim, h2_dim),
            nn.Tanh(),
            nn.Linear(h2_dim, h3_dim),
            nn.Tanh())
        self.net2 = nn.Linear(h3_dim, out_dim)

    def partial_forward(self, x):
        return self.net1(x)

    def forward(self, x):
        return self.net2(self.net1(x))

    def train(self, X, Y):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for _ in range(1000):
            optimizer.zero_grad()
            out = self(X).squeeze()
            loss = F.mse_loss(Y, out)
            loss.backward()
            optimizer.step()


def _expected_improvement(mean, sigma, min_val):
    mean = mean.numpy()
    sigma = sigma.numpy()
    min_val = min_val.numpy()
    dist = scipy.stats.norm(loc=0.0, scale=1.0)
    gamma = (min_val - mean) / sigma
    pdf = dist.pdf(x=gamma)
    cdf = scipy.stats.norm.cdf(x=gamma, loc=0., scale=1.)
    ei = (min_val - mean) * cdf + (sigma * pdf)
    return ei


class AcquisitonFunction:
    def __init__(self, aftype: str):
        if aftype == 'ei':
            self.af_func = _expected_improvement

    def __call__(self, *args, **kwargs):
        return self.af_func(*args, **kwargs)


class DNGO:
    def __init__(self, trial):
        self.trial = trial
        self.nn = SimpleNetwork(self.trial.num_params(), 1)
        self.acq_func = AcquisitonFunction('ei')
        self.train_x = []
        self.train_y = []

    def random_search(self, n_trial=3):
        trials = np.random.randint(0, len(self.trial), n_trial)
        for i in trials:
            t = self.trial.create_trainer(i)
            y = t.run()
            self.train_y.append(y)
        self.train_x = self.trial.get_delete_list()

    def bayes_search(self, n_trial=100):
        for i in range(n_trial):
            self.run(self.train_x, self.train_y)
            mean, var = self.predict(self.to_tensor(self.trial.remains()))
            acq = self.calc_acq_value(mean, var)
            next_sample = np.argmax(acq)
            t = self.trial.create_trainer(next_sample)
            y = t.run()
            print(f'acc = {y}')
            self.train_y.append(-y) # change to minimum problem
            self.train_x = self.trial.get_delete_list()

    def train(self, x, y):
        self.nn.train(x, y)
        bases = self.nn.partial_forward(x)
        params = scipy.optimize.fmin(self.calc_marginal_log_likelihood,
                                     torch.rand(2),
                                     args=(bases, x, y))
        return self.to_tensor(params)

    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, list):
            return torch.from_numpy(np.array(x)).float()
        return x

    def run(self, x, y):
        x = copy.deepcopy(x)
        x = self.to_tensor(x)
        y = self.to_tensor(y)
        self.norm_x, self.mean_x, self.std_x = normalize_col(x)
        self.norm_y, self.mean_y, self.std_y = normalize_1d(y)
        self.params = self.train(self.norm_x, self.norm_y)
        bases = self.nn.partial_forward(self.norm_x)
        self.calc_marginal_log_likelihood(self.params, bases, self.norm_x, self.norm_y)

    def calc_acq_value(self, mean, var):
        min_val = torch.min(self.norm_y)
        return self.acq_func(mean, var, min_val)

    def predict(self, x):
        _x = copy.deepcopy(x)
        _, beta = torch.exp(self.params).float()
        _x = (_x - self.mean_x) / self.std_x
        phi = self.nn.partial_forward(_x)
        mean = torch.matmul(phi, self.m)
        var = torch.diag(torch.matmul(torch.matmul(phi, self.K_inv), phi.t()) + 1 / beta)
        mean = mean * self.std_y + self.mean_y
        var = var * self.std_y ** 2
        #var = var.reshape(mean.shape[0], mean.shape[1])
        return mean.detach(), var.detach()

    def calc_marginal_log_likelihood(self, theta, phi=None, x=None, y=None):
        theta = torch.tensor(theta).float()
        alpha = torch.exp(theta[0])
        beta = torch.exp(theta[1])
        n_samples = x.shape[0]
        dim_features = x.shape[1]
        # phi dimenstion is coresponding to (B, D)
        # B is batch size. D is feature dimmension.

        # # calculate K matrix
        identity = torch.eye(phi.shape[1])
        phi_T = phi.transpose(1, 0)
        K = beta * torch.matmul(phi_T, phi) + alpha * identity

        # # calculate m
        K_inv = torch.inverse(K)
        m = beta * torch.matmul(K_inv, phi_T)
        y_tilde = self.calc_quadritc_prior(y)
        m = torch.matmul(m, y_tilde)

        self.m = m
        self.K_inv = K_inv
        m = torch.squeeze(m)
        # # calculate log_p
        mll = dim_features / 2. * np.log(alpha)
        mll += n_samples / 2. * np.log(beta)
        mll -= n_samples / 2. * np.log(2*math.pi)
        mll -= beta / 2. * torch.norm(y_tilde - torch.matmul(phi, m))
        mll -= alpha / 2. * m.dot(m)
        mll -= 0.5 * torch.log(torch.det(K))
        return -mll

    def calc_quadritc_prior(self, x):
        # is not implmented yet
        return x


class Net(nn.Module):
    def __init__(self, ch1, ck1, cs1, ch2, ck2, cs2, lh):
        super(Net, self).__init__()
        ch1 = int(ch1)
        ck1 = int(ck1)
        cs1 = int(cs1)
        ch2 = int(ch2)
        ck2 = int(ck2)
        cs2 = int(cs2)
        lh = int(lh)
        self.conv1 = nn.Conv2d(1, ch1, ck1, cs1)
        self.conv2 = nn.Conv2d(ch1, ch2, ck2, cs2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = exnn.Linear(lh)
        self.fc2 = exnn.Linear(10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Trainer:
    def __init__(self, model, lr, batchsize):
        batchsize = int(batchsize)
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr)
        self.batchsize = batchsize
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batchsize, shuffle=True)
        self.n_iteraterion = 100
        self.device = 'cuda'

    def run(self):
        self.model.train()
        self.model.to(self.device)
        correct = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            try:
                output = self.model(data)
            except Exception as e:
                print(str(e))
                return 0.0
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx == self.n_iteraterion:
                break
        return 100. * correct / (self.n_iteraterion * self.batchsize)


def trainer_factory(batchsize, lr, ch1, ck1, cs1, ch2, ck2, cs2, lh):
    try:
        m = Net(ch1, ck1, cs1, ch2, ck2, cs2, lh)
    except:
        m = None
    t = Trainer(m, lr, batchsize)
    return t


class Trial:
    def __init__(self):
        self.params = dict(
            batchsize=[64, 128, 256, 512, 1024],
            lr=[i * 1e-4 for i in range(10)],
            ch1=[16, 64, 128],
            ch2 = [16, 64, 128],
            ck1=[1, 2, 3],
            cs1=[1, 2],
            ck2=[1, 2, 3],
            cs2=[1, 2],
            lh=[92, 128, 256])
        self.params_n_list = []
        self.length = 1
        self.length_list = []
        for k in self.params:
            n = len(self.params[k])
            self.length *= n
            self.length_list.append(self.length)
            self.params_n_list.append(n)

        self.param_list = np.array(list(itertools.product(*list(self.params.values()))))
        self.delete_list = []

    def remains(self):
        return self.param_list

    def get_delete_list(self):
        return np.array(self.delete_list)

    def num_params(self):
        return len(self.params)

    def pop(self, idx):
        '''
        q = idx
        param = {}
        for k in self.params:
            n = len(self.params[k])
            idx = q % n
            q = q // n
            param[k] = self.params[k][idx]
        '''
        param = {}
        for j, k in enumerate(self.params):
            param[k] = self.param_list[idx, j]
        self.delete_list.append(self.param_list[idx])
        np.delete(self.param_list, idx)
        print('trial', param)
        return param

    def create_trainer(self, idx):
        return trainer_factory(**self.pop(idx))

    def __len__(self):
        return self.length


def main():
    print(len(Trial()))
    d = DNGO(Trial())
    return
    d.random_search()
    d.bayes_search()


if __name__ == '__main__':
    main()
