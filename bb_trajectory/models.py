from . import classification
import torch
import torch.nn
import sklearn.metrics
import numpy as np

class CNN1D(torch.nn.Module):

    training_history = None

    # Hyperparameters.
    num_layers = 2
    n_channels = 9
    kernel_size = 5
    batch_norm = True
    conv_dropout = 0.33
    max_pooling = True
    grow_batch_size = True
    initial_batch_size = 64
    initial_learning_rate = 0.0003
    learning_rate_decay = 0.98
    optimizer = "adam"

    def get_params(self, deep=True):
        return {"num_layers": self.num_layers,
                "n_channels": self.n_channels,
                "kernel_size": self.kernel_size,
                "batch_norm": self.batch_norm,
                "conv_dropout": self.conv_dropout,
                "max_pooling": self.max_pooling,
                "grow_batch_size": self.grow_batch_size,
                "initial_batch_size": self.initial_batch_size,
                "initial_learning_rate": self.initial_learning_rate,
                "learning_rate_decay": self.learning_rate_decay,
                "optimizer": self.optimizer
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def __init__(self, n_features, timesteps, num_layers=2, n_channels=9,\
                 kernel_size=5, conv_dropout=0.33, max_pooling=True, batch_norm=True,
                 initial_batch_size=64, grow_batch_size=True, initial_learning_rate=0.001, learning_rate_decay=0.98, train_epochs=100,
                 cuda=True, optimizer="adam"):
        super(CNN1D, self).__init__()

        self.num_layers = num_layers
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.conv_dropout = conv_dropout
        self.max_pooling = max_pooling
        self.grow_batch_size = grow_batch_size
        self.initial_batch_size = initial_batch_size
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.optimizer = optimizer

        self.convs = []
        self.n_features = n_features
        self.o_depth = timesteps
        self.use_cuda = cuda
        self.train_epochs = train_epochs
        
        channels = [n_features] + [int(n_channels)] * (int(num_layers))
        for i in range(len(channels) - 1):
            if i > 0 and self.batch_norm:
                self.convs.append(torch.nn.BatchNorm1d(channels[i]))
            self.convs.append(torch.nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, padding=kernel_size // 2))
            self.convs.append(torch.nn.SELU())
            if i > 0 and i < len(channels) - 1 and conv_dropout > 0.001:
                self.convs.append(torch.nn.AlphaDropout(p=conv_dropout))
            if max_pooling:
                self.convs.append(torch.nn.MaxPool1d(2))
                self.o_depth = self.o_depth // 2
        self.convs = torch.nn.Sequential(*self.convs)
        
        self.out_channels = channels[-1]
        
        last_conv_size = self.o_depth * self.out_channels
        denses = []
        denses.append(torch.nn.Linear(last_conv_size, 8))
        denses.append(torch.nn.SELU())
        denses.append(torch.nn.Linear(8, 1))
        denses.append(torch.nn.Sigmoid())
        
        self.denses = torch.nn.Sequential(*denses)
        
        if cuda:
            self.cuda()

    def forward(self, inputs, hidden=None, force=True, steps=0, verbose=False, cuda=None):
        if cuda is None:
            cuda = self.use_cuda

        batch_size = inputs.size(0)
        
        #print (inputs.size())
        inputs = self.convs(inputs)
        #print (inputs.size())
        inputs = inputs.view(batch_size, self.out_channels * self.o_depth)
        True
        inputs = self.denses(inputs)
        #print("ALL IS FINE.")
        #inputs = self.denses(inputs)
        return inputs

    def predict_proba(self, X, cuda=None):
        if cuda is None:
            cuda = self.use_cuda

        self.eval()
        
        def batch_iter(arr, batch_size=128):
            i = 0
            while i < arr.shape[0]:
                yield arr[i:(i + batch_size)]
                i += batch_size

        Y_predicted = []
        for batch in batch_iter(X):
            batch_X = torch.from_numpy(batch)
            batch_X = torch.autograd.Variable(batch_X)
            if cuda:
                batch_X = batch_X.cuda()
            y = self.forward(batch_X, cuda=cuda)
            if cuda:
                y = y.cpu()
            Y_predicted.append(y.data.numpy())

        Y_predicted = np.vstack(Y_predicted)
        # probas.
        return np.hstack((1.0 - Y_predicted, Y_predicted))

    def predict(self, *args, **kwargs):
        Y_probas = self.predict_proba(*args, **kwargs)
        return np.argmax(Y_probas, axis=1)

    def fit(self, X, y, test_X=None, test_y=None, clean=True, progress=None, checkpoint_path=None,
        optimizer=None, criterion=None, cuda=None, metric=None, show_graphs=False, figsize=(16, 2)):
        if cuda is None:
            cuda = self.use_cuda

        progress_wrapper = lambda x: x
        if progress is not None and "tqdm" in progress:
            import tqdm
            if progress == "tqdm":
                progress_wrapper = tqdm.tqdm
            elif progress == "tqdm_notebook":
                progress_wrapper = tqdm.tqdm_notebook
        else:
            progress = None
        epoch_iterator = progress_wrapper(list(range(self.train_epochs)))

        if optimizer is None:
            if self.optimizer == "adam":
                optimizer = torch.optim.Adam
            elif self.optimizer == "sgd":
                optimizer = torch.optim.SGD
            else:
                ValueError("Unknown optimizer: {}".format(self.optimizer))

        criterion = criterion() if not criterion is None else torch.nn.BCELoss()
        if cuda:
            criterion.cuda()

        if metric is None:
            metric = sklearn.metrics.f1_score

        if self.training_history is None or clean:
            self.training_history = {"loss": [], "train_score": [], "test_score": [], "test_score_best": [],
                                    "learning_rate": [], "batch_size": [], "checkpoint": []}


        batch_size = self.initial_batch_size or 64
        
        loss_step = 10
        test_score = 0.0
        best_score = 0.0
        y_predicted_train, y_predicted_test = None, None

        for epoch in epoch_iterator:
            running_loss = 0.0

            batch_size_epoch = 1.0
            if self.grow_batch_size:
                batch_size_epoch = 1 + (0.05 * min(epoch, 50))

            current_batch_size = int(batch_size * batch_size_epoch)
            current_learning_rate = ((self.initial_learning_rate / (batch_size_epoch * 0.5)) * (self.learning_rate_decay ** epoch))

            epoch_optimizer = optimizer(self.parameters(), lr=current_learning_rate)

            minibatch_iter = enumerate(classification.iterate_minibatches(X, y, batchsize=current_batch_size, shuffle=True))
            for minibatch_idx, (batch_X, batch_Y) in minibatch_iter:
                #print(batch_X.shape)
                #break
                batch_X = torch.from_numpy(batch_X)
                
                batch_X = torch.autograd.Variable(batch_X)
                if cuda:
                    batch_X = batch_X.cuda()

                Y_predicted = self.forward(batch_X, verbose=False, cuda=cuda)

                #batch_Y = onehot.transform(batch_Y)
                batch_Y = torch.from_numpy(batch_Y)
                batch_Y = torch.autograd.Variable(batch_Y.float())
                if cuda:
                    batch_Y = batch_Y.cuda()
                loss = criterion(Y_predicted, batch_Y)

                self.zero_grad()
                loss.backward()

                epoch_optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if progress is not None and (minibatch_idx % loss_step == (loss_step - 1)):
                    epoch_iterator.set_postfix({"Batch Size": "{:5d}".format(current_batch_size),
                                            "Test Score": test_score,
                                            "Test Score (Best)": best_score,
                                            })

            self.training_history["loss"].append(running_loss / X.shape[0])
            self.training_history["batch_size"].append(current_batch_size)
            self.training_history["learning_rate"].append(current_learning_rate)

            saved = False
            if test_X is not None:
                y_predicted_test = self.predict(test_X, cuda=cuda) > 0.5
                test_score = metric(test_y.flatten(), y_predicted_test.flatten())
                if test_score > best_score:
                    best_score = test_score
                    if epoch > 2 and checkpoint_path is not None:
                        torch.save(self, checkpoint_path)
                    saved = True
                    
                self.training_history["test_score"].append(test_score)
                self.training_history["test_score_best"].append(best_score)

                y_predicted_train = self.predict(X, cuda=cuda) > 0.5
                self.training_history["train_score"].append(metric(y.flatten(), y_predicted_train.flatten()))

                self.train()
            self.training_history["checkpoint"].append(saved)

        self.eval()
        
        if show_graphs:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(figsize[0], 2 * figsize[1]))
            ax.plot(self.training_history["loss"], "k:", label="Loss")
            ax.set_title("Training progress (final loss: {:7.5f})".format(self.training_history["loss"][-1]))
            ax.legend(loc="lower left")

            ax = ax.twinx()
            ax.plot(self.training_history["train_score"], "r", label="Train")
            ax.plot(self.training_history["test_score"], "g", label="Test")
            ax.plot(self.training_history["test_score_best"], "g:", alpha=0.5, label="Test (Best)")
            checkpoint_indices = np.where(self.training_history["checkpoint"])[0]
            checkpoint_scores = np.take(self.training_history["test_score_best"], checkpoint_indices)
            ax.plot(checkpoint_indices, checkpoint_scores, linestyle="None", marker=r"$\checkmark$", color="g", markersize=13, label="Checkpoint")
            ax.set_ylim(0,1)
            plt.legend(loc="center right")
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(self.training_history["learning_rate"], "k--", label="Learning rate")
            plt.legend(loc="center left")

            ax = ax.twinx()
            ax.plot(self.training_history["batch_size"], "b", label="Batch size")
            plt.legend(loc="center right")
            plt.tight_layout()
            plt.show()

            if test_X is not None:
                print("Test set report:")
                y_predicted_test = self.predict_proba(test_X, cuda=cuda)
                classification.display_classification_report(y_predicted_test, test_y, figsize=(2 * figsize[1], 2 * figsize[1]))
                print("Train set report:")
                y_predicted_train = self.predict_proba(X, cuda=cuda)
                classification.display_classification_report(y_predicted_train, y, figsize=(2 * figsize[1], 2 * figsize[1]))