import os
from itertools import cycle

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
import sklearn.metrics
from scipy import interp
from sklearn.preprocessing import OneHotEncoder


def make_cost_acc_plot(train_cost_list, valid_cost_list, train_acc_list, valid_acc_list, result_dir, metric_name="acc",
                       metric_show_name="Accuracy"):
    loss_path = os.path.join(result_dir, "loss.png")
    plt.plot(train_cost_list, 'k-', label='Train Set Cost')
    plt.plot(valid_cost_list, 'r-', label='Validation Set Cost')
    plt.title("Loss per Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='lower right')
    plt.savefig(loss_path)
    print(f"[SAVE] Cost figure in [ {loss_path} ] ")
    plt.clf()

    metric_path = os.path.join(result_dir, f"{metric_name}.png")
    plt.plot(train_acc_list, 'k-', label=f'Train Set {metric_show_name}')
    plt.plot(valid_acc_list, 'r-', label=f'Validation Set {metric_show_name}')
    plt.title(f'Train and Validation {metric_show_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_show_name)
    plt.legend(loc='lower right')
    plt.savefig(metric_path)
    print(f"[SAVE] {metric_show_name} figure in [ {metric_path} ] ")
    plt.clf()


def regularize_multitask_label(label):
    if len(label.shape) == 2:
        num_classes = int(np.max(label)+1)
        encoder = OneHotEncoder(n_values=num_classes)
        x = []
        for l in label:
            l_ = l.reshape((-1, 1))
            x.append(encoder.fit_transform(l_).toarray())
        return np.array(x)
    elif len(label.shape) == 3:
        return label
    else:
        pass
    return None


def regularize_multitask_score(score):
    if len(score.shape) == 2:
        score0 = 1-score
        score1 = score
        return np.stack((score0, score1), axis=-1)
    elif len(score.shape) == 3:
        return score
    else:
        pass
    return None


def make_multitask_auc_plot(true_label, pred_score, result_dir, plot_each_class=True):
    true_label = regularize_multitask_label(true_label)
    pred_score = regularize_multitask_score(pred_score)
    num_task = true_label.shape[1]
    for i in range(num_task):
        make_auc_plot(true_label[:, i, :], pred_score[:, i, :], result_dir, plot_each_class, postfix=str(i))


def make_auc_plot(true_label, pred_score, result_dir, plot_each_class=True, postfix=None):
    print(true_label.shape, pred_score.shape)
    n_classes = len(true_label[0])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pred_score = np.array(pred_score)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_label[:, i], pred_score[:, i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_label.ravel(), pred_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'darkred'])
    if plot_each_class:
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_curve')
    plt.legend(loc="lower right", fontsize=10)
    filename = os.path.join(result_dir, "auc.png") if postfix is None else os.path.join(result_dir, f"auc{postfix}.png")
    plt.savefig(filename)
    plt.clf()
    print(f"[SAVE] ROC/AUC in [ {filename} ] ")


def make_r2_plot(true_label, pred_score, result_dir, postfix=None):
    plt.figure()
    r2 = sklearn.metrics.r2_score(true_label, pred_score)
    mse = sklearn.metrics.mean_squared_error(true_label, pred_score)
    if len(true_label.shape) == 1:
        true_label = true_label[:, np.newaxis]
    x = true_label
    y = pred_score
    # Create linear regression object
    regr = LinearRegression()
    regr.fit(x, y)
    yp = regr.predict(x)
    print(f'Coefficients: \n{regr.coef_}\n'
          f'Mean squared error: {mse:.2f}\n'
          f'r2: {r2:.2f}\n')
    # Plot outputs
    plt.scatter(x, y,  color='black')
    plt.plot(x, yp, color='blue', linewidth=3)

    filename = os.path.join(result_dir, "r2.png") if postfix is None else os.path.join(result_dir, f"r2{postfix}.png")
    plt.savefig(filename)
    plt.clf()
    print(f"[SAVE] R2 plot in [ {filename} ] ")


def plot_cost(config, data, model, prefix=""):
    result_path = config["plot_path"]
    os.makedirs(result_path, exist_ok=True)
    training_acc = [el["training_accuracy"] for el in model.training_metrics_list]
    validation_acc = [el["validation_accuracy"] for el in model.validation_metrics_list]
    make_cost_acc_plot(model.training_cost_list, model.validation_cost_list, training_acc, validation_acc,
                       result_path+prefix)


def plot_auc(config, labels, pred_data, prefix=""):
    result_path = config["plot_path"]
    os.makedirs(result_path, exist_ok=True)
    if config["plot_multitask"]:
        make_multitask_auc_plot(labels, pred_data, result_path+prefix)
    else:
        make_auc_plot(labels, pred_data, result_path+prefix)


def plot_r2(config, labels, pred_data, prefix=""):
    result_path = config["plot_path"]
    os.makedirs(result_path, exist_ok=True)
    if config["plot_multitask"]:
        print("not supported")
    else:
        make_r2_plot(labels, pred_data, result_path+prefix)
