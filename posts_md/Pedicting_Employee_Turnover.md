<h1 style="font-family: Georgia; font-size: 3em; color:#2462C0; font-style:bold">
Predicting Employee Turnover
</h1>

<p align="center">
<img src = "../images/employee-turnover.png" style="height:300px; width:700px"><br>
</p>

<h2 style="font-family: Georgia; font-size: 2em; color:purple; font-style:bold">
Introduction
</h2>

**Employee turnover** refers to the percentage of workers who leave an organization and are replaced by new employees. It is very costly for organizations, where costs include but not limited to: separation, vacancy, recruitment, training and replacement. On average, organizations invest between four weeks and three months training new employees. This investment would be a loss for the company if the new employee decided to leave the first year. Furthermore, organizations such as consulting firms would suffer from deterioration in customer satisfaction due to regular changes in *Account Reps* and/or *Consultants* that would lead to loss of businesses with clients.

In this post, we’ll work on simulated HR data from [kaggle](https://www.kaggle.com/ludobenistant/hr-analytics-1) to build a classifier that helps us predict what kind of employees will be more likely to leave given some attributes. Such classifier would help an organization predict employee turnover and be pro-active in helping to solve such costly matter. We’ll restrict ourselves to use the most common classifiers: Random Forest, Gradient Boosting Trees, K-Nearest Neighbors, Logistic Regression and Support Vector Machine.

The data has 14,999 examples (samples). Below are the features and the definitions of each one:
- satisfaction_level: Level of satisfaction {0–1}.
- last_evaluationTime: Time since last performance evaluation (in years).
- number_project: Number of projects completed while at work.
- average_montly_hours: Average monthly hours at workplace.
- time_spend_company: Number of years spent in the company.
- Work_accident: Whether the employee had a workplace accident.
- left: Whether the employee left the workplace or not {0, 1}.
- promotion_last_5years: Whether the employee was promoted in the last five years.
- sales: Department the employee works for.
- salary: Relative level of salary {low, medium, high}.

Source code that created this post can be found [here](https://nbviewer.jupyter.org/github/ImadDabbura/blog-posts/blob/master/notebooks/Employee-Turnover.ipynb).

<h2 style="font-family: Georgia; font-size: 2em; color:purple; font-style:bold">
Data Preprocessing
</h2>

Let’s take a look at the data (check if there are missing values and the data type of each features):

```
# Load the data
df = pd.read_csv(“data/HR_comma_sep.csv”)
# Check both the datatypes and if there is missing values
print(“\033[1m” + “\033[94m” + “Data types:\n” + 11 * “-”)
print(“\033[30m” + “{}\n”.format(df.dtypes))
print(“\033[1m” + “\033[94m” + “Sum of null values in each column:\n” + 35 * “-”)
print(“\033[30m” + “{}”.format(df.isnull().sum()))
df.head()
```

<p align="center">
<img src = "../posts_images/employee_turnover/data_types.PNG" style="height:700px; width:600px"><br>
</p>
<caption><center>Data overview</center></caption>

Since there are no missing values, we do not have to do any imputation. However, there are some data preprocessing needed:
1. Change **sales** feature name to **department**.
2. Convert **salary** into ordinal categorical feature since there is intrinsic order between: low, medium and high.
3. Create dummy features from **department** feature and drop the first one to avoid linear dependency where some learning algorithms may struggle.

```
# Rename sales feature into department
df = df.rename(columns={"sales": "department"})
# Map salary into integers
salary_map = {"low": 0, "medium": 1, "high": 2}
df["salary"] = df["salary"].map(salary_map)
# Create dummy variables for department feature
df = pd.get_dummies(df, columns=["department"], drop_first=True)
```

The data is now ready to be used for modeling. The final number of features are now 17.

<h2 style="font-family: Georgia; font-size: 2em; color:purple; font-style:bold">
Modeling
</h2>

Let’s first take a look at the proportion of each class to see if we’re dealing with balanced or imbalanced data, since each one has its own set of tools to be used when fitting classifiers.

```
# Get number of positve and negative examples
pos = df[df["left"] == 1].shape[0]
neg = df[df["left"] == 0].shape[0]
print("Positive examples = {}".format(pos))
print("Negative examples = {}".format(neg))
print("Proportion of positive to negative examples = {:.2f}%".format((pos / neg) * 100))
sns.countplot(df["left"])
plt.xticks((0, 1), ["Didn't leave", "Left"])
plt.xlabel("Left")
plt.ylabel("Count")
plt.title("Class counts");
```

<p align="center">
<img src = "../posts_images/employee_turnover/class_counts.PNG" style="height:500px; width:500px"><br>
</p>
<caption><center>Class counts</center></caption>

As the graph shows, we have an imbalanced dataset. As a result, when we fit classifiers on such datasets, we should use metrics other than accuracy when comparing models such as *f1-score* or *AUC* (area under ROC curve). Moreover, class imbalance influences a learning algorithm during training by making the decision rule biased towards the majority class by implicitly learns a model that optimizes the predictions based on the majority class in the dataset. There are three ways to deal with this issue:
1. Assign a larger penalty to wrong predictions from the minority class.
2. Upsampling the minority class or downsampling the majority class.
3. Generate synthetic training examples.

Nonetheless, there is no definitive guide or best practices to deal with such situations. Therefore, we have to try them all and see which one works best for the problem on hand. We’ll restrict ourselves to use the first two, i.e, assign larger penalty to wrong predictions from the minority class using `class_weight` in classifiers that allows us do that and evaluate upsampling/downsampling on the training data to see which gives higher performance.
First, split the data into training and test sets using 80/20 split; 80% of the data will be used to train the models and 20% to test the performance of the models. Second, Upsample the minority class and downsample the majority class. For this data set, positive class is the minority class and negative class is the majority class.

```
# Convert dataframe into numpy objects and split them into
# train and test sets: 80/20
X = df.loc[:, df.columns != "left"].values
y = df.loc[:, df.columns == "left"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)
# Upsample minority class
X_train_u, y_train_u = resample(X_train[y_train == 1],
                                y_train[y_train == 1],
                                replace=True,
                                n_samples=X_train[y_train == 0].shape[0],
                                random_state=1)
X_train_u = np.concatenate((X_train[y_train == 0], X_train_u))
y_train_u = np.concatenate((y_train[y_train == 0], y_train_u))
# Downsample majority class
X_train_d, y_train_d = resample(X_train[y_train == 0],
                                y_train[y_train == 0],
                                replace=True,
                                n_samples=X_train[y_train == 1].shape[0],
                                random_state=1)
X_train_d = np.concatenate((X_train[y_train == 1], X_train_d))
y_train_d = np.concatenate((y_train[y_train == 1], y_train_d))
print("Original shape:", X_train.shape, y_train.shape)
print("Upsampled shape:", X_train_u.shape, y_train_u.shape)
print("Downsampled shape:", X_train_d.shape, y_train_d.shape)
```

- Original shape: (11999, 17) (11999,)
- Upsampled shape: (18284, 17) (18284,)
- Downsampled shape: (5714, 17) (5714,)

I don’t think we need to apply dimensionality reduction such as PCA because: 1) We want to know the importance of each feature in determining who will leave versus who will not (inference). 2) Dimension of the data set is decent (17 features). However, it’s good to see how many principal components needed to explain 90%, 95% and 99% of the variation in the data.

```
# Build PCA using standarized trained data
pca = PCA(n_components=None, svd_solver="full")
pca.fit(StandardScaler().fit_transform(X_train))
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(12, 6))
plt.bar(range(1, 18), pca.explained_variance_ratio_, align="center",
        color='red', label="Individual explained variance")
plt.step(range(1, 18), cum_var_exp, where="mid", label="Cumulative explained variance")
plt.xticks(range(1, 18))
plt.legend(loc="best")
plt.xlabel("Principal component index", {"fontsize": 14})
plt.ylabel("Explained variance ratio", {"fontsize": 14})
plt.title("PCA on training data", {"fontsize": 16});
```
<p align="center">
<img src = "../posts_images/employee_turnover/PCA.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>PCA</center></caption>

Looks like it needs 14, 15 and 16 principal components to capture 90%, 95% and 99% of the variation in the data respectively. In other words, this means that the data is already in a good space since eigenvalues are very close to each other and gives further evidence that we don’t need to compress the data.
The methodology that we’ll follow when building the classifiers goes as follows:
1. Build a pipeline that handles all the steps when fitting the classifier using scikit-learn’s `make_pipeline` which will have two steps:
I. Standardizing the data to speed up convergence and make all features on the same scale.
II. The classifier (`estimator`) we want to use to fit the model.
2. Use `GridSearchCV` to tune hyperparameters using 10-folds cross validation. We can use `RandomizedSearchCV` which is faster and may outperform `GridSearchCV` especially if we have more than two hyperparameters and the range for each one is very big; however, `GridSearchCV` will work just fine since we have only two hyperparameters and descent range.
3. Fit the model using training data.
4. Plot both confusion matrix and ROC curve for the best estimator using test data.
Repeat the above steps for *Random Forest, Gradient Boosting Trees, K-Nearest Neighbors, Logistic Regression and Support Vector Machine*. Next, pick the classifier that has the highest cross validation f1 score. Note: some of the hyperparameter ranges will be guided by the paper [Data-driven Advice for Applying Machine Learning to Bioinformatics Problems](https://arxiv.org/pdf/1708.05070.pdf).

<h3 style="font-family: Georgia; font-size: 1.5em; color:purple; font-style:bold">
Random Forest
</h3>

First, we will start by fitting a Random Forest classifier using *unsampled, upsampled and downsampled* data. Second, we will evaluate each method using cross validation (CV) f1-score and pick the one with the highest CV f1-score. Finally, we will use that method to fit the rest of the classifiers.
The only hyperparameters we’ll tune are:
- `max_feature`: how many features to consider randomly on each split. This will help avoid having few strong features to be picked on each split and let other features have the chance to contribute. Therefore, predictions will be less correlated and the variance of each tree will decrease.
- `min_samples_leaf`: how many examples to have for each split to be a final leaf node.
Random Forest is an ensemble model that has multiple trees (`n_estimators`). The final prediction would be a weighting average (regression) or mode (classification) of the predictions from all estimators. Note: a high number of trees doesn't cause overfitting.

```
# Build random forest classifier
methods_data = {"Original": (X_train, y_train),
                "Upsampled": (X_train_u, y_train_u),
                "Downsampled": (X_train_d, y_train_d)}
for method in methods_data.keys():
    pip_rf = make_pipeline(StandardScaler(),
                           RandomForestClassifier(n_estimators=500,
                                                  class_weight="balanced",
                                                  random_state=123))
    
    hyperparam_grid = {
        "randomforestclassifier__n_estimators": [10, 50, 100, 500],
        "randomforestclassifier__max_features": ["sqrt", "log2", 0.4, 0.5],
        "randomforestclassifier__min_samples_leaf": [1, 3, 5],
        "randomforestclassifier__criterion": ["gini", "entropy"]}
    
    gs_rf = GridSearchCV(pip_rf,
                         hyperparam_grid,
                         scoring="f1",
                         cv=10,
                         n_jobs=-1)
    
    gs_rf.fit(methods_data[method][0], methods_data[method][1])
    
    print("\033[1m" + "\033[0m" + "The best hyperparameters for {} data:".format(method))
    for hyperparam in gs_rf.best_params_.keys():
        print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_rf.best_params_[hyperparam])
        
    print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_rf.best_score_) * 100))
```

<p align="center">
<img src = "../posts_images/employee_turnover/rf_hyperparam.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>Random Forest hyperparameter tuning results</center></caption>

Upsampling yielded the highest CV f1-score with 99.8%. Therefore, we’ll be using the upsampled data to fit the rest of the classifiers. The new data now has 18,284 examples: 50% belonging to the positive class, and 50% belonging to the negative class.

Let’s refit the Random Forest with Upsampled data using best hyperparameters tuned above and plot confusion matrix and ROC curve using test data.

<p align="center">
<img src = "../posts_images/employee_turnover/rf_roc.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>Random Forest</center></caption>

<h3 style="font-family: Georgia; font-size: 1.5em; color:purple; font-style:bold">
Gradient Boosting Trees
</h3>

Gradient Boosting trees are the same as Random Forest except for:
- It starts with small tree and start learning from grown trees by taking into account the residual of grown trees.
- More trees can lead to overfitting; opposite to Random Forest.
Therefore, we can think of each tree as a weak learner. The two other hyperparameters than `max_features` and `n_estimators` that we're going to tune are:
- `learning_rate`: rate the tree learns, the slower the better.
- `max_depth`: number of split each time a tree is growing which limits the number of nodes in each tree.

Let’s fit GB classifier and plot confusion matrix and ROC curve using test data.

```
# Build Gradient Boosting classifier
pip_gb = make_pipeline(StandardScaler(),
                       GradientBoostingClassifier(loss="deviance",
                                                  random_state=123))
hyperparam_grid = {"gradientboostingclassifier__max_features": ["log2", 0.5],
                   "gradientboostingclassifier__n_estimators": [100, 300, 500],
                   "gradientboostingclassifier__learning_rate": [0.001, 0.01, 0.1],
                   "gradientboostingclassifier__max_depth": [1, 2, 3]}
gs_gb = GridSearchCV(pip_gb,
                      param_grid=hyperparam_grid,
                      scoring="f1",
                      cv=10,
                      n_jobs=-1)
gs_gb.fit(X_train, y_train)
print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_gb.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_gb.best_params_[hyperparam])
print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_gb.best_score_) * 100))
```

<p align="center">
<img src = "../posts_images/employee_turnover/gb_hyperparam.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>Gradient Boosting Trees hyperparameter tuning results</center></caption><br>

<p align="center">
<img src = "../posts_images/employee_turnover/gb_roc.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>Gradient Boosting Trees</center></caption>

<h3 style="font-family: Georgia; font-size: 1.5em; color:purple; font-style:bold">
K-Nearest Neighbors
</h3>

KNN is called a lazy learning algorithm because it doesn’t learn or fit any parameter. It takes `n_neighbors` points from the training data closest to the point we're interested to predict it's class and take the mode (majority vote) of the classes for the neighboring point as its class. The two hyperparameters we're going to tune are:
- `n_neighbors`: number of neighbors to use in prediction.
- `weights`: how much weight to assign neighbors based on:
- “uniform”: all neighboring points have the same weight.
- “distance”: use the inverse of euclidean distance of each neighboring point used in prediction.

Let’s fit KNN classifier and plot confusion matrix and ROC curve.

```
# Build KNN classifier
pip_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
hyperparam_range = range(1, 20)
gs_knn = GridSearchCV(pip_knn,
                      param_grid={"kneighborsclassifier__n_neighbors": hyperparam_range,
                                  "kneighborsclassifier__weights": ["uniform", "distance"]},
                      scoring="f1",
                      cv=10,
                      n_jobs=-1)
gs_knn.fit(X_train, y_train)
print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_knn.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_knn.best_params_[hyperparam])
print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_knn.best_score_) * 100))
```

<p align="center">
<img src = "../posts_images/employee_turnover/knn_hyperparam.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>K-Nearest Neighbors hyperparameter tuning results</center></caption><br>

<p align="center">
<img src = "../posts_images/employee_turnover/knn_roc.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>K-Nearest Neighbors</center></caption>

<h3 style="font-family: Georgia; font-size: 1.5em; color:purple; font-style:bold">
Logistic Regression
</h3>
For logistic regression, we’ll tune three hyperparameters:
`penalty`: type of regularization, L2 or L1 regularization.
- `C`: the opposite of regularization of parameter λλ. The higher C the less regularization. We'll use values that cover the full range between unregularized to fully regularized where model is the mode of the examples' label.
- `fit_intercept`: whether to include intercept or not.

We won’t use any non-linearities such as polynomial features.

```
# Build logistic model classifier
pip_logmod = make_pipeline(StandardScaler(),
                           LogisticRegression(class_weight="balanced"))
hyperparam_range = np.arange(0.5, 20.1, 0.5)
hyperparam_grid = {"logisticregression__penalty": ["l1", "l2"],
                   "logisticregression__C":  hyperparam_range,
                   "logisticregression__fit_intercept": [True, False]
                  }
gs_logmodel = GridSearchCV(pip_logmod,
                           hyperparam_grid,
                           scoring="accuracy",
                           cv=2,
                           n_jobs=-1)
gs_logmodel.fit(X_train, y_train)
print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_logmodel.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_logmodel.best_params_[hyperparam])
print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_logmodel.best_score_) * 100))
```

<p align="center">
<img src = "../posts_images/employee_turnover/logmodel_hyperparam.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>Logistic Regression hyperparameter tuning results</center></caption><br>

<p align="center">
<img src = "../posts_images/employee_turnover/logmodel_roc.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>Logistic Regression</center></caption>

<h3 style="font-family: Georgia; font-size: 1.5em; color:purple; font-style:bold">
Support Vector Machine
</h3>

SVM is comutationally very expensive to tune it’s hyperparameters for two reasons:
1. With big datasets, it becomes very slow.
2. It has good number of hyperparameters to tune that takes very long time to tune on a CPU.

Therefore, we’ll use recommended hyperparameters’ values from the paper we mentioned before that showed to yield the best performane on Penn Machine Learning Benchmark 165 datasets. The hyperparameters that we usually look to tune are:
- `C, gamma, kernel, degree` and `coef0`

```
# Build SVM classifier
clf_svc = make_pipeline(StandardScaler(),
                        SVC(C=0.01,
                            gamma=0.1,
                            kernel="poly",
                            degree=5,
                            coef0=10,
                            probability=True))
clf_svc.fit(X_train, y_train)
svc_cv_scores = cross_val_score(clf_svc,
                                X=X_train,
                                y=y_train,
                                scoring="f1",
                                cv=10,
                                n_jobs=-1)
# Print CV
print("\033[1m" + "\033[94m" + "The 10-folds CV f1-score is: {:.2f}%".format(
       np.mean(svc_cv_scores) * 100))
The 10-folds CV f1-score is: 96.38%
```

**The 10-folds CV f1-score is: 96.38%**

<p align="center">
<img src = "../posts_images/employee_turnover/svc_roc.PNG" style="height:400px; width:700px"><br>
</p>
<caption><center>Support Vector Machine</center></caption>

<h2 style="font-family: Georgia; font-size: 2em; color:purple; font-style:bold">
Conclusion
</h2>

Let’s conclude by printing out the test accuracy rates for all classifiers we’ve trained so far and plot ROC curves. Then we will pick the classifier that has the highest area under ROC curve.

<p align="center">
<img src = "../posts_images/employee_turnover/overall.PNG" style="height:600px; width:700px"><br>
</p>
<caption><center>Comparing ROC curves for all classifiers</center></caption>

Even though Random Forest and Gradient Boosting Trees have almost equal AUC, Random Forest has higher accuracy rate and an f1-score with 99.27% and 99.44% respectively. Therefore, we safely say Random Forest outperforms the rest of the classifiers. Let’s have a look of feature importances from Random Forest classifier.

<p align="center">
<img src = "../posts_images/employee_turnover/feature_importance.PNG" style="height:500px; width:700px"><br>
</p>
<caption><center>Random Forest feature importance</center></caption>

Looks like the five most important features are:
- satisfaction_level
- time_spend_company
- average_montly_hours
- number_project
- lats_evaluation

The take home message is the following:
- When dealing with imbalanced classes, accuracy is not a good method for model evaluation. AUC and f1-score are examples of metrics we can use.
- Upsampling/downsampling, data synthetic and using balanced class weights are good strategies to try to improve the accuracy of a classifier for imbalanced classes datasets.
- `GridSearchCV` helps tune hyperparameters for each learning algorithm. `RandomizedSearchCV` is faster and may outperform `GridSearchCV` especially when we have more than two hyperparameters to tune.
- Principal Component Analysis (PCA) isn’t always recommended especially if the data is in a good feature space and their eigen values are very close to each other.
- As expected, ensemble models outperforms other learning algorithms in most cases.