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
        sub_targets = targets[excerpt] if targets is not None else None
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

def display_classification_report(Y_predicted, Y_ground_truth, has_proba=True, roc_auc=True, figsize=None):
    if roc_auc and not has_proba:
        raise ValueError("ROC AUC curve needs probability scores.")

    Y_probas = Y_predicted
    if has_proba:
        Y_classes = np.argmax(Y_probas, axis=1)
    else:
        Y_classes = Y_probas
    
    print(sklearn.metrics.classification_report(Y_ground_truth, Y_classes))

    if roc_auc:
        fpr, tpr, _ = sklearn.metrics.roc_curve(Y_ground_truth, Y_probas[:, 1])
        plt.figure(figsize=figsize)

        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.axis("equal")
        plt.show()

