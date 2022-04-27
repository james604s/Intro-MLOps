import numpy as np 
import pandas as pd 
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix
from sklearn.model_selection import KFold 
import mlflow
import mlflow.sklearn

print("Numpy: {}".format(np.__version__))
print("Pandas: {}".format(pd.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("seaborn: {}".format(sns.__version__))
print("Scikit-Learn: {}".format(sklearn.__version__))
print("MLFlow: {}".format(mlflow.__version__))

data_path = "dataset/creditcard.csv"
df = pd.read_csv(data_path)
df = df.drop("Time", axis=1)

df.head()

#################################### Data Processing ####################################

"""
Randomly sampling 50% of all the normal data points in the data frame and picking out all of the anomalies from the data frame as separate data frames. Then, you print the shapes of both data sets. As you can see, the normal points massively outnumber the anomaly points
"""
normal = df[df.Class == 0].sample(frac=0.5, random_state=2020).reset_index(drop=True) 
anomaly = df[df.Class == 1]
print(f"Normal: {normal.shape}")
print(f"Anomaly: {anomaly.shape}")


"""
Partitioning the normal and anomaly data frames separately into train, test, and validation splits. Initially, 20% of
the normal and anomaly points are used as the test split. From
the remaining 80% of data, 25% of that train split is used as the validation split, meaning the validation split is 20% of the original data. This leaves the final training split at 60% of the original data. In the end, the train-test-validate split has a 60-20-20 ratio, respectively
"""
normal_train, normal_test = train_test_split(normal, test_size = 0.2, random_state = 2020)
anomaly_train, anomaly_test = train_test_split(anomaly, test_size = 0.2, random_state = 2020)
normal_train, normal_validate = train_test_split(normal_train,test_size = 0.25, random_state = 2020)
anomaly_train, anomaly_validate = train_test_split(anomaly_train, test_size = 0.25, random_state = 2020)

"""
Creating the respective x and y splits of the training, testing, and validation sets by concatenating the respective normal and anomaly sets. You drop Class from the x-sets because it would be cheating otherwise to give it the label directly. You are trying to get the model to learn the labels by reading the x-data, not learn how to read the Class column in the x-data
"""
x_train = pd.concat((normal_train, anomaly_train))
x_test = pd.concat((normal_test, anomaly_test))
x_validate = pd.concat((normal_validate, anomaly_validate))
y_train = np.array(x_train["Class"])
y_test = np.array(x_test["Class"])
y_validate = np.array(x_validate["Class"])
x_train = x_train.drop("Class", axis=1)
x_test = x_test.drop("Class", axis=1)
x_validate = x_validate.drop("Class", axis=1)

print("Training sets:\nx_train: {} \ny_train:{}".format(x_train.shape, y_train.shape))
print("\nTesting sets:\nx_test: {} \ny_test:{}".format(x_test.shape, y_test.shape))
print("\nValidation sets:\nx_validate: {} \ny_validate: {}".format(x_validate.shape, y_validate.shape))

"""
Fitting the scaler on the superset of normal and anomaly points after dropping Class to scale the x-sets
"""
scaler = StandardScaler()
scaler.fit(pd.concat((normal, anomaly)).drop("Class", axis=1))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validate = scaler.transform(x_validate)


#################################### Training and Evaluating with MLFlow####################################

def train(sk_model, x_train, y_train): 
    """
    Defining the train function to better organize the code. Additionally, you are defining a training accuracy metric that will be logged by MLFlow
    """
    sk_model = sk_model.fit(x_train, y_train)
    train_acc = sk_model.score(x_train, y_train)
    mlflow.log_metric("train_acc", train_acc)
    print(f"Train Accuracy: {train_acc:.3%}")

def evaluate(sk_model, x_test, y_test):
    """
    A function to calculate the evaluation metrics for the AUC score and accuracy. Plots for the confusion matrix and the ROC curve are generated, and both the metrics and the graphs are logged to MLFlow
    """
    eval_acc = sk_model.score(x_test, y_test)
    preds = sk_model.predict(x_test)
    auc_score = roc_auc_score(y_test, preds)
    # MLFlow 指標紀錄
    mlflow.log_metric("eval_acc", eval_acc)
    mlflow.log_metric("auc_score", auc_score)
    print(f"Auc Score: {auc_score:.3%}")
    print(f"Eval Accuracy: {eval_acc:.3%}")
    roc_plot = plot_roc_curve(sk_model, x_test, y_test,
    name='Scikit-learn ROC Curve')
    plt.savefig("sklearn_roc_plot.png")
    plt.show()
    plt.clf()
    conf_matrix = confusion_matrix(y_test, preds)
    ax = sns.heatmap(conf_matrix, annot=True,fmt='g') 
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix") 
    plt.savefig("sklearn_conf_matrix.png")
    # MLFlow 圖紀錄
    mlflow.log_artifact("sklearn_roc_plot.png")
    mlflow.log_artifact("sklearn_conf_matrix.png")


#################################### Logging and Viewing MLFlow Runs####################################
sk_model = LogisticRegression(random_state=None, max_iter=400, solver='newton-cg')
mlflow.set_experiment("scikit_learn_experiment") 
#If that name does not exist, MLFlow will create a new one under that name and put the run there.
with mlflow.start_run():
    train(sk_model, x_train, y_train)
    evaluate(sk_model, x_test, y_test)
    mlflow.sklearn.log_model(sk_model, "log_reg_model")
    print("Model run: ", mlflow.active_run().info.run_uuid)
mlflow.end_run()

#mlflow ui -p 1234

#################################### Loading a Logged Model####################################
# loaded_model = mlflow.sklearn.load_model("runs:/YOUR_RUNID_HERE/log_reg_model")
loaded_model = mlflow.sklearn.load_model("runs:/f23c101d414e43be877210a6f954a1cd/log_reg_model")
loaded_model.score(x_test, y_test)


#################################### Model Validation (Parameter Tuning) with MLFlow####################################
#Parameter Tuning – Broad Search
anomaly_weights = [1, 5, 10, 15]
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=2020)

mlflow.set_experiment("sklearn_creditcard_broad_search") 
logs = []
for f in range(len(anomaly_weights)):
    fold = 1
    accuracies = []
    auc_scores= []
    for train, test in kfold.split(x_validate, y_validate):
        with mlflow.start_run():
            weight = anomaly_weights[f] 
            mlflow.log_param("anomaly_weight", weight)
            class_weights= {
                0: 1,
                1: weight 
            }
            sk_model = LogisticRegression(random_state=None, max_iter=400,
                                                solver='newton-cg',
                                                class_weight=class_weights).fit(x_validate[train],y_validate[train])
            for h in range(40): 
                print('-', end="")
                print(f"\nfold {fold}\nAnomaly Weight: {weight}")
                train_acc = sk_model.score(x_validate[train],y_validate[train])
                mlflow.log_metric("train_acc", train_acc)
                eval_acc = sk_model.score(x_validate[test],y_validate[test])
                preds = sk_model.predict(x_validate[test])
                mlflow.log_metric("eval_acc", eval_acc)
            #Here is some more of the code. Make sure this all aligns with the code from above.
            try:
                auc_score = roc_auc_score(y_validate[test], preds)
            except:
                auc_score = -1
                mlflow.log_metric("auc_score", auc_score)
                print("AUC: {}\neval_acc: {}".format(auc_score,
                eval_acc))
                accuracies.append(eval_acc)
                auc_scores.append(auc_score)
                log = [sk_model, x_validate[test],
                y_validate[test], preds]
                logs.append(log)
                mlflow.sklearn.log_model(sk_model,
                f"anom_weight_{weight}_fold_{fold}")
                fold = fold + 1
                mlflow.end_run()
print("\nAverages: ")
print("Accuracy: ", np.mean(accuracies))
print("AUC: ", np.mean(auc_scores))
print("Best: ")
print("Accuracy: ", np.max(accuracies))
print("AUC: ", np.max(auc_scores))


#metrics."auc_score" >= 0.90
#metrics."auc_score" >= 0.90 AND parameters.anomaly_weight = "5"
#parameters.anomaly_weight = "1"


#Parameter Tuning – Guided Search
#The best overall performances were achieved by anomaly weights 10 and 15, but it seems to be on an upward trend the higher up you go with the anomaly weight. Now that you know this, let’s try another validation run with a broader range of anomaly weights to try.
anomaly_weights = [10, 50, 100, 150, 200]
mlflow.set_experiment("sklearn_creditcard_guided_search")

#parameters.anomaly_weight = "10"