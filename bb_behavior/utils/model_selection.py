from .. import trajectory
import numpy as np
import pandas
import sklearn.metrics
import matplotlib.pyplot as plt

# source: https://stackoverflow.com/a/38176770
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if targets is not None:
        assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            
        if targets is None:
            yield inputs[excerpt]
        else:
            sub_targets = targets[excerpt]
            yield inputs[excerpt], sub_targets

def optimization_objective_function(make_model_fun, train_model_fun=None, datareader=None, scorer=None, X=None, Y=None, groups=None, *args, **kwargs):
    if scorer is None:
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)

    if datareader is None:
        datareader = trajectory.features.DataReader.from_XY(X, Y, groups=groups, features=None)

    def _make_model_fun():
        return make_model_fun(*args, **kwargs)

    scores = datareader.cross_validate(_make_model_fun, train_model_fun=train_model_fun, scorer=scorer, *args, **kwargs)
    
    # We minimize the score.
    return 1.0 - np.mean(scores)


def minimize_objective(fun, arg_space, max_evals=100, *args, **kwargs):
    import hyperopt
    trials = hyperopt.Trials()
    best = hyperopt.fmin(fun, arg_space, algo=hyperopt.tpe.suggest, max_evals=max_evals, trials=trials)
    best_set = hyperopt.space_eval(arg_space, best)

    trials_df = []
    for trial in trials.trials:
        row = dict(score = trial["result"]["loss"])
        vals = {k:v[0] for k,v in trial["misc"]["vals"].items()}
        vals = hyperopt.space_eval(arg_space, vals)

        for key in vals:
            row[key] = float(vals[key])
        trials_df.append(row)
    trials_df = pandas.DataFrame(trials_df)
    
    return best_set, trials_df

def plot_roc_curve(Y_true, Y_predicted, ax=None):

    fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(Y_true, Y_predicted)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    lw = 2
    ax.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.set_aspect("equal")
    
    if fig:
        plt.show()

    return fpr, tpr, roc_thresholds

def plot_thresholds_curve(Y_true, Y_predicted, tpr=None, thresholds=None, ax=None):
    if tpr is None:
        _, tpr, thresholds = sklearn.metrics.roc_curve(Y_true, Y_predicted)
    
    percentage_thresholds = np.linspace(0.0, 1.0, num=thresholds.shape[0])
    percentages = [(np.sum(Y_predicted >= t) / Y_predicted.shape[0]) for t in percentage_thresholds]

    precision = [sklearn.metrics.precision_score(Y_true, Y_predicted >= t) for t in percentage_thresholds]
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Influence of threshold on results")

    f_scores = [[sklearn.metrics.fbeta_score(Y_true, Y_predicted >= t, f_beta) for t in percentage_thresholds]
                for f_beta in (0.5, 1.0, 2.0)]
    ax.fill_between(percentage_thresholds, f_scores[0], f_scores[-1], color="gray", alpha=0.5)
    ax.plot(percentage_thresholds, f_scores[1], c="gray", linestyle=":", label="F-Score", alpha=0.5)
        

    ax.plot(percentage_thresholds, percentages, "k-", label="Positive rate")
    ax.plot(percentage_thresholds, precision, "r", label="Precision (PPV)")
    ax.plot(thresholds, tpr, "g--", label="Hit rate (TPR; Recall)")
    ax.plot(thresholds, 1.0 - tpr, "b:", label="Miss rate (FNR; 1.0 - TPR)")
    ax.set_ylabel("Fraction of data")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend()
    ax.set_xlabel("Threshold")
    ax.set_aspect("equal")
    if fig is not None:
        plt.show()

    return tpr, thresholds

def plot_precision_recall_curve(Y_true, Y_predicted, ax=None):

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(Y_true, Y_predicted)
    precision = precision[::-1]
    recall = recall[::-1]
    thresholds = thresholds[::-1]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    lw = 2
    ax.plot(recall, precision, color='darkorange', lw=lw, label='Precision/Recall curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision/Recall curve')
    ax.set_aspect("equal")
    if fig is not None:
        plt.show()

    return precision, recall, thresholds

def display_classification_report(Y_predicted, Y_ground_truth, has_proba=True, roc_auc=True, precision_recall=True, plot_thresholds=True, figsize=None):
    n_threshold_curves = int(roc_auc) + int(precision_recall) + int(plot_thresholds)
    if n_threshold_curves > 0 and not has_proba:
        raise ValueError("Threshold curves need probability scores.")

    Y_probas = Y_predicted
    if has_proba:
        if len(Y_probas.shape) > 1:
            Y_classes = np.argmax(Y_probas, axis=1)
            Y_probas = Y_probas[:, 1]
        else:
            Y_classes = Y_probas > 0.5
    else:
        Y_classes = Y_probas
    
    print(sklearn.metrics.classification_report(Y_ground_truth, Y_classes))

    if n_threshold_curves > 0:
        fig, axes = plt.subplots(1, n_threshold_curves, figsize=(7 * n_threshold_curves, 7))
        ax_idx = 0
        for (check, fun) in zip((roc_auc, precision_recall, plot_thresholds), (plot_roc_curve, plot_precision_recall_curve, plot_thresholds_curve)):
            if check:
                fun(Y_ground_truth, Y_probas, ax=axes[ax_idx])
                ax_idx += 1
        fig.tight_layout()
        plt.show()

