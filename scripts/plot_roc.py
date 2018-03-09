import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, roc_curve


def plot_conf_matrix_and_roc(estimator, X, y, figure_size=(16, 6)):
    """
    Plot both confusion matrix and ROC curce on the same figure.

    Parameters:
    -----------
    estimator : sklearn.estimator
        model to use for predicting class probabilities.
    X : array_like
        data to predict class probabilities.
    y : array_like
        true label vector.
    figure_size : tuple (optional)
        size of the figure.

    Returns:
    --------
    plot : matplotlib.pyplot
        plot confusion matrix and ROC curve.
    """
    # Compute tpr, fpr, auc and confusion matrix
    fpr, tpr, thresholds = roc_curve(y, estimator.predict_proba(X)[:, 1])
    auc = roc_auc_score(y, estimator.predict_proba(X)[:, 1])
    conf_mat_rf = confusion_matrix(y, estimator.predict(X))

    # Define figure size and figure ratios
    plt.figure(figsize=figure_size)
    gs = GridSpec(1, 2, width_ratios=(1, 2))

    # Plot confusion matrix
    ax0 = plt.subplot(gs[0])
    ax0.matshow(conf_mat_rf, cmap=plt.cm.Reds, alpha=0.2)

    for i in range(2):
        for j in range(2):
            ax0.text(x=j, y=i, s=conf_mat_rf[i, j], ha="center", va="center")
    plt.title("Confusion matrix", y=1.1, fontdict={"fontsize": 20})
    plt.xlabel("Predicted", fontdict={"fontsize": 14})
    plt.ylabel("Actual", fontdict={"fontsize": 14})


    # Plot ROC curce
    ax1 = plt.subplot(gs[1])
    ax1.plot(fpr, tpr, label="auc = {:.3f}".format(auc))
    plt.title("ROC curve", y=1, fontdict={"fontsize": 20})
    ax1.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False positive rate", fontdict={"fontsize": 16})
    plt.ylabel("True positive rate", fontdict={"fontsize": 16})
    plt.legend(loc="lower right", fontsize="medium");


def plot_roc(estimators, X, y, figure_size=(16, 6)):
    """
    Plot both confusion matrix and ROC curce on the same figure.

    Parameters:
    -----------
    estimators : dict
        key, value for model name and sklearn.estimator to use for predicting
        class probabilities.
    X : array_like
        data to predict class probabilities.
    y : array_like
        true label vector.
    figure_size : tuple (optional)
        size of the figure.

    Returns:
    --------
    plot : matplotlib.pyplot
        plot confusion matrix and ROC curve.
    """
    plt.figure(figsize=figure_size)
    for estimator in estimators.keys():
        # Compute tpr, fpr, auc and confusion matrix
        fpr, tpr, thresholds = roc_curve(y, estimators[estimator].predict_proba(X)[:, 1])
        auc = roc_auc_score(y, estimators[estimator].predict_proba(X)[:, 1])

        # Plot ROC curce
        plt.plot(fpr, tpr, label="{}: auc = {:.3f}".format(estimator, auc))
        plt.title("ROC curve", y=1, fontdict={"fontsize": 20})
        plt.legend(loc="lower right", fontsize="medium")

    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate", fontdict={"fontsize": 16})
    plt.ylabel("True positive rate", fontdict={"fontsize": 16});


def plot_roc_and_pr_curves(models, X_train, y_train, X_valid, y_valid, roc_title, pr_title, labels):
    """
    Plot roc and PR curves for all models.
    
    Arguments
    ---------
    models : list
        list of all models.
    X_train : list or 2d-array
        2d-array or list of training data.
    y_train : list
        1d-array or list of training labels.
    X_valid : list or 2d-array
        2d-array or list of validation data.
    y_valid : list
        1d-array or list of validation labels.
    roc_title : str
        title of ROC curve.
    pr_title : str
        title of PR curve.
    labels : list
        label of each model to be displayed on the legend.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if not isinstance(X_train, list):
        for i, model in enumerate(models):
            model_fit = model.fit(X_train, y_train)
            model_probs = model.predict_proba(X_valid)[:, 1:]
            model_preds = model.predict(X_valid)
            model_auc_score = roc_auc_score(y_valid, model_probs)
            # model_f1_score = f1_score(y_valid, model_preds)
            fpr, tpr, _ = roc_curve(y_valid, model_probs)
            precision, recall, _ = precision_recall_curve(y_valid, model_probs)
            axes[0].plot(fpr, tpr, label=f"{labels[i]}, auc = {model_auc_score:.3f}")
            axes[1].plot(recall, precision, label=f"{labels[i]}")
    else:
        for i, model in enumerate(models):
            model_fit = model.fit(X_train[i], y_train[i])
            model_probs = model.predict_proba(X_valid[i])[:, 1:]
            model_preds = model.predict(X_valid[i])
            model_auc_score = roc_auc_score(y_valid[i], model_probs)
            # model_f1_score = f1_score(y_valid[i], model_preds)
            fpr, tpr, _ = roc_curve(y_valid[i], model_probs)
            precision, recall, _ = precision_recall_curve(y_valid[i], model_probs)
            axes[0].plot(fpr, tpr, label=f"{labels[i]}, auc = {model_auc_score:.3f}")
            axes[1].plot(recall, precision, label=f"{labels[i]}")
    axes[0].legend(loc="lower right")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title(roc_title)
    axes[1].legend()
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title(pr_title)
    plt.tight_layout()