from ..utils import model_selection
from ..trajectory.models import Fixup
import torch
import torch.nn
import math
import numpy as np
import sklearn.metrics


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class ConvNet(torch.nn.Module):
    
    def make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding=None, dropout=False):
        if padding is None:
            padding = kernel_size // 2
        components = [
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if dropout and dropout > 0.0:
            components.append(torch.nn.Dropout2d(dropout))
        components += [
            torch.nn.SELU(),
            Fixup()
         ]
        return torch.nn.Sequential(*components)
            
    def __init__(self, input_shape, n_classes=2, use_cuda=True, train_epochs=10, optimizer="adam",
                initial_learning_rate=0.01, class_weights=None, fixup_init=True, n_input_channels=1,
                learning_rate_decay=0.95, dropout=0.1):
        super(ConvNet, self).__init__()
        self.use_cuda = use_cuda
        self.train_epochs = train_epochs
        self.optimizer = optimizer
        self.initial_learning_rate = initial_learning_rate
        self.training_history = None
        self.class_weights = class_weights
        self.n_classes = n_classes
        self.learning_rate_decay = learning_rate_decay

        layers = []
        
        out_shape = input_shape
        sizes = (n_input_channels, 64, 32, 32, 8, n_classes)
        strides = (1, 2, 1, 2, 1, 1)
        dropouts = np.array([0, 0, 1, 0, 0, 0]) * dropout
        for idx in range(1, len(sizes)):
            stride = strides[idx]
            layers.append(self.make_conv_layer(sizes[idx - 1], sizes[idx],
                                kernel_size=5, stride=stride,
                                dropout=dropouts[idx]))
            out_shape = out_shape[0] // stride, out_shape[1] // stride
        layers.append(Flatten())
        
        layers.append(torch.nn.Linear(out_shape[0] * out_shape[1] * sizes[-1], n_classes))
        layers.append(torch.nn.Softmax(dim=1))
        
        self.layers = torch.nn.Sequential(*layers)

        if use_cuda:
            self.cuda()

        if fixup_init:
            layers = []
            def gather_layers(l):
                if type(l) == torch.nn.Linear:
                    layers.append(l)
            self.apply(gather_layers)
            m = len(layers)
            for idx, l in enumerate(layers[::-1]):
                if idx == 0:
                    l.weight.data.zero_()
                    l.bias.data.zero_()
                else:
                    l.weight.data.normal_(0, math.sqrt(2 / idx) * (m ** -0.5))


    def forward(self, inputs, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda
        batch_size = inputs.size(0)
        inputs = self.layers(inputs)
        return inputs
    
    def predict_proba(self, X, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda

        self.eval()
        
        def batch_iter(arr, batch_size=128):
            i = 0
            while i < arr.shape[0]:
                yield arr[i:(i + batch_size)]
                i += batch_size

        Y_predicted = []
        with torch.no_grad():
            for batch in batch_iter(X):
                batch_X = torch.from_numpy(batch)
                batch_X = torch.autograd.Variable(batch_X)
                if use_cuda:
                    batch_X = batch_X.cuda()
                y = self.forward(batch_X, use_cuda=use_cuda)
                if use_cuda:
                    y = y.cpu()
                Y_predicted.append(y.data.numpy())

        Y_predicted = np.vstack(Y_predicted)
        return Y_predicted

    def predict_proba_from_dataloader(self, dataloader, use_cuda=None, return_true_y=False):
        if use_cuda is None:
            use_cuda = self.use_cuda
        
        self.eval()

        all_predicted = []
        all_y = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_X, batch_Y = self.split_batch_from_generator(batch, use_cuda=use_cuda)
                Y_predicted = self.forward(batch_X, use_cuda=use_cuda)
                if use_cuda:
                    Y_predicted = Y_predicted.cpu()
                Y_predicted = Y_predicted.numpy()
                all_predicted.append(Y_predicted)

                if return_true_y:
                    y = batch_Y
                    if use_cuda:
                        y = y.cpu()
                    all_y.append(y.numpy())
        all_predicted = np.concatenate(all_predicted)
        if not return_true_y:
            return all_predicted
        all_y = np.concatenate(all_y)
        return all_predicted, all_y

    def predict(self, *args, **kwargs):
        Y_probas = self.predict_proba(*args, **kwargs)
        return np.argmax(Y_probas, axis=1)

    def split_batch_from_generator(self, batch, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda

        batch_X, batch_Y = batch
        batch_X = torch.autograd.Variable(batch_X)
        if batch_Y is not None:
            batch_Y = torch.autograd.Variable(batch_Y.float())
        if use_cuda:
            batch_X = batch_X.cuda()
            if batch_Y is not None:
                batch_Y = batch_Y.cuda()
        return batch_X, batch_Y
        
    def get_score_on_generator(self, generator, metric, use_cuda=None):
        test_Y = []
        predicted_Y = []
        for batch_idx, batch in enumerate(generator):
            batch_X, batch_Y = self.split_batch_from_generator(batch, use_cuda=use_cuda)
            Y_predicted = self.forward(batch_X, use_cuda=use_cuda)
            predicted_Y.append(Y_predicted.cpu().data.numpy()[:, 1])
            test_Y.append(batch_Y.cpu().data.numpy())

        test_Y = np.concatenate(test_Y).flatten()
        predicted_Y = np.concatenate(predicted_Y).flatten()
        test_score = None
        if metric is not None:
            test_score = metric(test_Y, predicted_Y > 0.5)
        return test_Y, predicted_Y, test_score

    def fit(self, train_generator, test_generator, clean=True, progress=None, checkpoint_path=None,
        optimizer=None, criterion=None, use_cuda=None, metric=None, show_graphs=False, figsize=(16, 2),
        lr_warmup_epochs=5, label_smoothing=0.1):
        if use_cuda is None:
            use_cuda = self.use_cuda

        progress_wrapper = lambda x, **y: x
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

        if metric is None:
            metric = sklearn.metrics.f1_score

        if self.training_history is None or clean:
            self.training_history = {"loss": [], "train_score": [], "test_score": [], "test_score_best": [],
                                    "learning_rate": [], "batch_size": [], "checkpoint": []}

        criterion = criterion() if not criterion is None else torch.nn.BCELoss()
        if use_cuda:
            criterion.cuda()
            
        loss_step = 10
        test_score = 0.0
        best_score = 0.0
        
        batch_size = None
        total_samples = 0
        for epoch in epoch_iterator:
            running_loss = 0.0
            batch_Y_onehot = torch.autograd.Variable(torch.FloatTensor(train_generator.batch_size, self.n_classes))
            if use_cuda:
                batch_Y_onehot = batch_Y_onehot.cuda()
            current_learning_rate = self.initial_learning_rate * (self.learning_rate_decay ** epoch)
            if lr_warmup_epochs and (epoch < lr_warmup_epochs):
                current_learning_rate = (epoch + 1) * current_learning_rate / lr_warmup_epochs

            epoch_optimizer = optimizer(self.parameters(), lr=current_learning_rate)

            for batch_idx, batch in enumerate(progress_wrapper(train_generator, leave=False)):
                
                batch_X, batch_Y = self.split_batch_from_generator(batch, use_cuda=use_cuda)
                Y_predicted = self.forward(batch_X, use_cuda=use_cuda)
                
                loss = None
                if batch_Y is not None:
                    if self.class_weights is not None:
                        weights = np.ones(shape=(batch_Y.shape[0]), dtype=np.float32)
                        for label, weight in self.class_weights.items():
                            weights[batch_Y.cpu().data.numpy() == label] = weight
                        weights = torch.from_numpy(weights)
                        criterion.weights = weights
                    batch_Y_onehot.data.zero_()
                    batch_Y_onehot.data.scatter_(1, batch_Y.data.type(torch.cuda.LongTensor).view(-1, 1), 1)

                    if label_smoothing > 0.0:
                        correct_labels_idx = batch_Y_onehot == 1.0
                        batch_Y_onehot[correct_labels_idx] -= label_smoothing
                        batch_Y_onehot[~correct_labels_idx] += label_smoothing / (self.n_classes - 1)

                    loss = criterion(Y_predicted, batch_Y_onehot)

                self.zero_grad()
                if loss is not None:
                    loss.backward()
                    running_loss += loss.item()
                    del loss

                epoch_optimizer.step()

                batch_size = batch_X.shape[0]
                total_samples += batch_size
                
                if progress is not None and (batch_idx % loss_step == (loss_step - 1)):
                    epoch_iterator.set_postfix({"Batch Size": "{:5d}".format(batch_size),
                                            "Test Score": test_score,
                                            "Test Score (Best)": best_score,
                                            })

            self.training_history["loss"].append(running_loss / total_samples)
            self.training_history["batch_size"].append(batch_size)
            self.training_history["learning_rate"].append(current_learning_rate)

            saved = False
            
            if test_generator is not None:
                self.eval()
                _, _, test_score = self.get_score_on_generator(test_generator, use_cuda=use_cuda, metric=metric) 

                if test_score > best_score:
                    best_score = test_score
                    if epoch > 2 and checkpoint_path is not None:
                        torch.save(self, checkpoint_path)
                    saved = True
                    
                self.training_history["test_score"].append(test_score)
                self.training_history["test_score_best"].append(best_score)

                #y_predicted_train = self.predict(X, use_cuda=use_cuda) > 0.5
                #self.training_history["train_score"].append(metric(y.flatten(), y_predicted_train.flatten()))

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

            if test_generator is not None:
                print("Test set report:")
                y, y_hat, score = self.get_score_on_generator(test_generator, use_cuda=use_cuda, metric=None)
                model_selection.display_classification_report(y_hat, y, figsize=(2 * figsize[1], 2 * figsize[1]))
                print("Train set report:")
                y, y_hat, score = self.get_score_on_generator(train_generator, use_cuda=use_cuda, metric=None)
                model_selection.display_classification_report(y_hat, y, figsize=(2 * figsize[1], 2 * figsize[1]))
            