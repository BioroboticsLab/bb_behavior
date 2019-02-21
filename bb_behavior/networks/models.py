from ..utils import iterate_minibatches

from collections import defaultdict
import math
import numpy as np
import pandas as pd
import sklearn

import torch
import torch.nn
import torch.nn.functional

class AttentionRegression(torch.nn.Module):
    
    def __init__(self, n_features_X, n_features_N, h_neighbours=12, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.n_features_X = n_features_X
        self.n_features_N = n_features_N
        self.f = torch.nn.Sequential(*[
                torch.nn.Linear(n_features_X + n_features_N, h_neighbours),
                torch.nn.Tanh(),
                torch.nn.Linear(h_neighbours, 1),
                torch.nn.Sigmoid()
                ])
        self.g = torch.nn.Sequential(*[
                torch.nn.Linear(n_features_X + n_features_N, 1),
                ])

        self.f_min = (+np.inf,) * (n_features_X + n_features_N)
        self.f_max = (-np.inf,) * (n_features_X + n_features_N)
    def forward(self, X):
        embedded_X = []
        for S in X:
            f_x, neighbours = S
            f_x = torch.autograd.Variable(torch.from_numpy(f_x))
            n_neighbours = len(neighbours)
            f_x_batch_expanded = f_x.expand(n_neighbours, 1)
            neighbours = np.array(neighbours, dtype=np.float32) # N x 2
            assert neighbours.shape[0] == n_neighbours
            neighbours = torch.from_numpy(neighbours)
            all_data = torch.cat((f_x_batch_expanded, neighbours), dim=1) # N x 3
            if self.training:
                all_data_numpy = all_data.detach().numpy()
                f_min, f_max = all_data_numpy.min(axis=0), all_data_numpy.max(axis=0)
                self.f_min = tuple(map(min, zip(self.f_min, f_min)))
                self.f_max = tuple(map(max, zip(self.f_max, f_max)))
            all_data = torch.autograd.Variable(all_data)
            
            f = self.f(all_data) # N x 3 -> N x 1
            attention_vector = torch.nn.functional.softmax(f, dim=0)
            
            neighbours = torch.autograd.Variable(neighbours)
            embedding = attention_vector * neighbours
            embedding = torch.sum(embedding, dim=0) # 1 x 2
            embedded = torch.cat((f_x, embedding), dim=0) # 1 x 3
            embedded_X.append(embedded)
            
        embedded_X = torch.cat(embedded_X).view(-1, self.n_features_X + self.n_features_N) # N x 3
        return self.g(embedded_X)
            
    def fit(self, X, y, weight_gain=0.001, batchsize=None, progress="tqdm", plot_loss=True):
        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight, gain=weight_gain)
                m.bias.data.fill_(0.01)
        self.apply(init_weights)

        progress_wrapper = lambda x: x
        if progress == "tqdm":
            from tqdm import tqdm
            progress_wrapper = tqdm
        elif progress == "tqdm_notebook":
            from tqdm import tqdm_notebook
            progress_wrapper = tqdm_notebook
        elif progress is not None:
            progress_wrapper = progress

        optimizer = torch.optim.RMSprop(self.parameters())
        criterion = torch.nn.MSELoss()
        self.losses = defaultdict(list)
        stop = False
        trange = progress_wrapper(range(100))
        for epoch in trange:
            if stop:
                break
            epoch_batchsize = batchsize
            if batchsize == "auto":
                epoch_batchsize = (1 + max(0, epoch - 5)) * 64
            elif not batchsize:
                epoch_batchsize = len(X)
            gen = iterate_minibatches(X, y, batchsize=epoch_batchsize, N=len(X))
            for X, Y in gen:
                self.train()
                self.zero_grad()
                
                Y = torch.Tensor(Y)
                Y_hat = self(X).flatten()
                loss = criterion(Y_hat, Y)
                loss.backward()
                optimizer.step()

                Y_hat = Y_hat.detach().numpy()
                if np.any(pd.isnull(Y_hat)):
                    print("Oh, no!")
                    print(Y_hat.flatten())
                    print(Y.flatten())
                    stop = True
                    break

                loss = math.sqrt(float(loss.data.numpy()))
                self.losses["loss"].append(loss)
                r2 = sklearn.metrics.r2_score(Y, Y_hat)
                self.losses["r2"].append(r2)
                if progress:
                    trange.set_postfix(loss="{:3.5f}".format(loss), r2="{:3.5f}".format(r2))

        if plot_loss:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(20,5))
            ax.plot(self.losses["loss"], "k--", label="Loss")
            plt.legend(loc="lower left")
            ax.twinx().plot(np.log(-(np.array(self.losses["r2"]) - 1.0)), label="-logR2")
            plt.legend(loc="upper right")
            plt.show()
        
    def predict(self, X):
        self.eval()
        Y_hat = self(X)
        return list(np.array(Y_hat.detach().numpy()).flatten())

    def plot_f(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        N = 2000
        N_inputs = len(self.f_min)
        fig, axes = plt.subplots(N_inputs, 1, figsize=(20, 3 * N_inputs))

        defaults = np.zeros(shape=(N, N_inputs), dtype=np.float32)
        for idx in range(N_inputs):
            defaults[:, idx] = np.random.uniform(self.f_min[idx], self.f_max[idx], size=N)
        F = self.f.forward
        
        for idx in range(N_inputs):
            ax = axes[idx]
            X = defaults.copy()
            X[:, idx] = np.linspace(self.f_min[idx], self.f_max[idx], num=N)
            Fy = F(torch.from_numpy(X)).detach().numpy().flatten()
            sns.regplot(X[:, idx], Fy, lowess=True, ax=ax)
            ax.set_title("Feature #{}".format(idx+1))
        plt.tight_layout()
        plt.show()