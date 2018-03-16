"""
Builing ensemble classifer using keras and sklearn.
"""
import numpy as np
from keras import models, layers, optimizers

# Build base learners and meta learner models
# Build nn model with keras (base learner)
def build_nn():
    nn_model = models.Sequential()
    nn_model.add(layers.Dense(128, activation="relu", input_shape=(18,)))
    nn_model.add(layers.Dropout(0.5))
    nn_model.add(layers.Dense(128, activation="relu"))
    nn_model.add(layers.Dense(1, activation="sigmoid"))

    nn_model.compile(optimizer=optimizers.SGD(lr=0.01),
                     loss="binary_crossentropy",
                     metrics=["acc"])
    return nn_model

# Build xgb model (base learner)
def build_xgb():
    model = xgb.XGBClassifier(objective="binary:logistic",
                              learning_rate=0.1,
                              n_estimators=100,
                              max_depth=3,
                              random_state=123)
    return model

# Build SVM model (base learner)
def build_svm():
    model = SVC(gamma=0.1,
                C=0.01,
                kernel="poly",
                degree=3,
                coef0=10.0,
                probability=True)
    return model

# Build logistic regression model (meta learner)
def build_logreg():
    model = LogisticRegression(penalty="l1",
                               C=50,
                               fit_intercept=True)
#     model = RandomForestClassifier(100)
    return model


def train_base_learner(base_learners, X_train, y_train, models_scores, cv=10, seed=123):
    # Initialize meta features and base learners trained
    meta_features = np.zeros((X_res.shape[0], len(base_learners)))

    # Split the data
    skf = StratifiedKFold(n_splits=cv, random_state=seed)
    k = 1
    print(f"Training using CV starts...")
    for train_index, valid_index in skf.split(X_res, y_res):
        train_data = X_res[train_index, :]
        train_labels = y_res[train_index]
        valid_data = X_res[valid_index]
        valid_labels = y_res[valid_index]

        i = 0
        for name, classifier in base_learners.items():
            # For nn model
            if "nn_" in name:
                classifier.fit(train_data, train_labels,
                               batch_size=64,
                               epochs=100,
                               verbose=0,
                               class_weight={0:1, 1: 4})
                model_preds = classifier.predict_proba(valid_data)
            # For all other models
            else:
                classifier.fit(train_data, train_labels)
                model_preds = classifier.predict_proba(valid_data)[:, 1]
            meta_features[valid_index, i] = model_preds.ravel()
            # Compute auc score on held out fold
            auc_score = roc_auc_score(valid_labels, model_preds.ravel())
            models_scores[name].append(auc_score)
        k += 1
    print(f"Done.")
    for name in models_scores:
        print(
            f"{name} {cv}-folds cv average AUC = {np.mean(models_scores[name]):.3f}")

    # Refit all the base learners on full training data
    print("Training using full training data starts...")
    base_learners_trained = {}
    for name, classifier in base_learners.items():
        # For nn
        if "nn_" in name:
            classifier.fit(X_res, y_res,
                           batch_size=64,
                           epochs=100,
                           verbose=0,
                           class_weight={0: 1, 1: 4})
        # For all other models
        else:
            classifier.fit(X_res, y_res)
        # Update base learners dictionary with fitted model
        base_learners_trained[name] = classifier
    print(f"Done.")
    return base_learners_trained, meta_features


def train_meta_learner(meta_learner, meta_features, y_train):
    # Fit meta learner on meta features
    model = meta_learner.fit(meta_features, y_res)
    return model


def predict_base_learners(base_learners, X_test):
    probs_matrix = np.zeros((X_test.shape[0], len(base_learners)))
    i = 0
    for name, classifier in base_learners.items():
        # For nn model
        if "nn_" in name:
            probs_matrix[:, i] = classifier.predict_proba(X_test).ravel()
        # For all other models
        else:
            probs_matrix[:, i] = classifier.predict_proba(X_test)[:, 1]
        i +=1
    return probs_matrix


def predict_meta_learner(meta_learner, meta_features):
    return meta_learner.predict_proba(meta_features)[:, 1]


def ensemble_avg(probs_matrix):
    return probs_matrix.mean(axis=1)