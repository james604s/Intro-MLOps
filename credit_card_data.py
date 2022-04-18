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
from pylab import rcParams


rcParams['figure.figsize'] = 14, 8

data_path = "dataset/creditcard.csv"
df = pd.read_csv(data_path)

df.head()

df.describe()

df.info()

df.shape

#Normal Data and Fraudulent Data
anomalies = df[df['Class'] == 1]
normal = df[df['Class'] == 0]
print(f"Anomalies: {anomalies.shape}")
print(f"Normal: {normal.shape}")


class_counts = pd.value_counts(df['Class'], sort = True) 
class_counts.plot(kind = 'bar', rot=0)
plt.title("Class Distribution")
plt.xticks(range(2), ["Normal", "Anomaly"]) 
plt.xlabel("Label")
plt.ylabel("Counts")

anomalies['Amount'].describe()

def plot_histogram(df, bins, column, log_scale=False): 
    bins = 100
    anomalies = df[df.Class == 1]
    normal = df[df.Class == 0]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True) 
    fig.suptitle(f'Counts of {column} by Class')
    ax1.hist(anomalies[column], bins = bins, color="red")
    ax1.set_title('Anomaly')
    ax2.hist(normal[column], bins = bins, color="orange")
    ax2.set_title('Normal')
    plt.xlabel(f'{column}') 
    plt.ylabel('Count')
    if log_scale:
        plt.yscale('log')
        plt.xlim((np.min(df[column]), np.max(df[column])))
        plt.show()

def plot_scatter(df, x_col, y_col, sharey = False): 
    anomalies = df[df.Class == 1]
    normal = df[df.Class == 0]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=sharey)
    fig.suptitle(f'{y_col} over {x_col} by Class')
    ax1.scatter(anomalies[x_col], anomalies[y_col], color='red') 
    ax1.set_title('Anomaly')
    ax2.scatter(normal[x_col], normal[y_col], color='orange')
    ax2.set_title('Normal')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

"""
plotting values for Amount by Class 
A scatterplot of data values in the data frame encompassing all the data values. 
The plotted columns are Amount on the x-axis and Class on the y-axis
"""
plt.scatter(df.Amount, df.Class) 
plt.title("Transaction Amounts by Class") 
plt.ylabel("Class")
plt.yticks(range(2), ["Normal", "Anomaly"]) 
plt.xlabel("Transaction Amounts ($)")
plt.show()

"""
A histogram of counts for data values organized into intervals in the column Amount in the data frame. 
The number of bins is 100, meaning the interval of each bar in the histogram is the range of the data in the column Amount divided by the number of bins
"""
bins = 100
plot_histogram(df, bins, "Amount", log_scale=True)

"""
A histogram of just the values in the anomaly data frame for the column Amount. 
The number of bins is also 100 here, as it will be for the rest of the examples
"""
plt.hist(anomalies.Amount, bins = bins, color="red")
plt.show()

"""
A scatterplot for values in the data frame df with data in the column 
Time on the x-axis and data in the column Class in the y-axis
"""
plt.scatter(df.Time, df.Class) 
plt.title("Transactions over Time by Class") 
plt.ylabel("Class")
plt.yticks(range(2), ["Normal", "Anomaly"])
plt.xlabel("Time (s)")
plt.show()


"""
This graph isn’t very informative, 
but it does tell you that fraudulent transactions are pretty spread out over the entire timeline. 
Once again, let’s use the plotter functions to get an idea of the counts:

Using the plot_scatter() function to plot data values for the columns Time on the x-axis and Amount on the y-axis in the df data frame
"""
plot_scatter(df, "Time", "Amount")

"""
Using the plot_histogram() function to plot data values for the column Time in the df data frame
"""
plot_histogram(df, bins, "Time")

"""
F1-19 Using the plot_histogram() function to plot the data in the column V1 in df
"""
plot_histogram(df, bins, "V1")

"""
F1-20 Using the plot_scatter() function to plot the values in the columns Amount on the x-axis and V1 on the y-axis in df
"""
plot_scatter(df, "Amount", "V1", sharey=True)

"""
F1-21 Using the plot_scatter() function to plot the values in the columns Time on the x-axis and V1 on the y-axis in df
"""
plot_scatter(df, "Time", "V1", sharey=True)

"""
F1-22~26
"""
for f in range(1, 29):
    print(f'V{f} Counts') 
    plot_histogram(df, bins, f'V{f}')

"""
F1-27 , 28
"""
for f in range(1, 29):
    print(f'V{f} vs Time')
    plot_scatter(df, "Time", f'V{f}', sharey=True)


"""
F1-29~31
"""
for f in range(1, 29):
    print(f'Amount vs V{f}')
    plot_scatter(df, f'V{f}', "Amount", sharey=True)


#################################### Build Model Data Processing####################################
# Check version
print("numpy: {}".format(np.__version__))
print("pandas: {}".format(pd.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("seaborn: {}".format(sns.__version__))
print("sklearn: {}".format(sklearn.__version__))


data_path = "dataset/creditcard.csv"
df = pd.read_csv(data_path)

normal = df[df['Class'] == 0].sample(frac=0.5, random_state=2020). reset_index(drop=True)
anomaly = df[df['Class'] == 1]
"""
First, you will split the data into train and test data, keeping the normal points and anomalies separate. 
To do this, you will use the train_test_ split() function from scikit-learn. Commonly passed parameters are
• x: The x set you want to split up
• y: The y set you want to split up corresponding to the x set
• test_size: The proportion of data in x and y that you want to randomly sample for the test set.
"""
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 2020)
#plitting the normal and anomaly data frames into train and test subsets. The respective test sets comprise 20% of the original sets
#0.8 0.2
normal_train, normal_test = train_test_split(normal, test_size= 0.2, random_state = 2020)
anomaly_train, anomaly_test = train_test_split(anomaly,test_size = 0.2, random_state = 2020)

#You create train and validate splits from the training data. 
# You have chosen to make the validation set comprise 25% of the respective original training sets. 
# As these original training sets themselves comprise of 80% of the original normal and anomaly data frames, 
# the respective validation splits are 20% (0. 25 * 0.8) of their original normal and anomaly data frames. 
# And so, the final training split also becomes 60% of the original, as 0.75 * 0.8 = 0.6
# 0.6 0.2 0.2
normal_train, normal_validate = train_test_split(normal_train,test_size = 0.25, random_state = 2020)
anomaly_train, anomaly_validate = train_test_split(anomaly_train, test_size = 0.25, random_state = 2020)

"""
To create your final training, testing, and validation sets, 
you have to concatenate the respective normal and anomaly data splits.

First, you define x_train, x_test, and x_validate:
"""
x_train = pd.concat((normal_train, anomaly_train))
x_test = pd.concat((normal_test, anomaly_test))
x_validate = pd.concat((normal_validate, anomaly_validate))

"""
Next, you define y_train, y_test, and y_validate:
"""
y_train = np.array(x_train["Class"])
y_test = np.array(x_test["Class"])
y_validate = np.array(x_validate["Class"])

"""
Finally, you have to drop the column Class in the x sets since 
it would defeat the purpose of teaching the model how to learn 
what makes up a normal and a fraudulent transaction if you gave it the label directly:
"""
x_train = x_train.drop("Class", axis=1)
x_test = x_test.drop("Class", axis=1)
x_validate = x_validate.drop("Class", axis=1)

#Let’s get the shapes of the sets you just created:

print("Training sets:\nx_train: {} y_train: {}".format(x_train.shape, y_train.shape))
print("\nTesting sets:\nx_test: {} y_test: {}".format(x_test.shape, y_test.shape))
print("\nValidation sets:\nx_validate: {} y_validate:{}".format(x_validate.shape, y_validate.shape))

"""
In more detail, the model will have a hard time optimizing the cost function and may take many more steps to converge, if it is able to do so at all.
And so it is better to scale everything down by normalizing the data. You will be using scikit-learn’s StandardScaler, which normalizes all of the data such that the mean is 0 and the standard deviation is 1.
Here is the code to standardize your data:
"""

scaler = StandardScaler()
scaler.fit(pd.concat((normal, anomaly)).drop("Class", axis=1))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validate = scaler.transform(x_validate)

"""
It is important to note that you are fitting the scaler on the entire data frame so that it standardizes all of your data in the same way. 
This is to ensure the best results since you don’t want to standardize x_train, x_test, and x_validate in their own ways since it would create discrepancies in
the data and would be problematic for the model. 
Of course, once you’ve deployed the model and start receiving new data, you would still standardize it using the scaler from the training process, but this new data could possibly come from a slightly different distribution than your training data. This would especially be the case if trends start shifting - this new standardized data could possibly lead to a tougher time for the model since 
it wouldn’t fit very well in the distribution that the model trained on.
"""

#################################### Model Training ####################################
sk_model = LogisticRegression(random_state=None, max_iter=400, solver='newton-cg').fit(x_train, y_train)

#################################### Model Evalutaion ####################################
"""
You can now look at accuracy and AUC scores. 
First, you find the accuracy using the built-in score function of the model:

AUC is usually a better metric since it better explains the performance of the model. 
The general gist of it is that a model that perfectly classifies every point correctly will have an AUC score of 100%.

However, the AUC score would show a value much lower at around 0.5. 
An AUC of 0.5 means that the model knows nothing and is practically just guessing randomly, 
but in this case, it means the model only ever predicts “normal” for any point it sees. 
In other words, it hasn’t actually learned much of anything if it doesn’t know how to predict an anomaly.
It’s also worth mentioning that AUC isn’t the sole metric by which one should base the worthiness of a model, 
since context matters. In this case, normal points far outnumber anomalies, 
so accuracy is a relatively poor metric to solely judge model performance on. 
AUC scores in this case would reflect the mode’s performance well, 
but it’s also possible to get higher AUC scores but lower accuracy scores. 
"""
eval_acc = sk_model.score(x_test, y_test)

"""
Now, let’s get the predictions and calculate the AUC score:
"""
preds = sk_model.predict(x_test)
auc_score = roc_auc_score(y_test, preds)

print(f"Auc Score: {auc_score:.3%}")
print(f"Eval Accuracy: {eval_acc:.3%}")

roc_plot = plot_roc_curve(sk_model, x_test, y_test,name='Scikit-learn ROC Curve')


"""
What’s basically happening is that scikit-learn takes in the model and the evaluation set to dynamically generate the curve as it predicts on the test sets. 
The metrics you see on the axes are derived from how correctly the model predicts each of the values. 
The “true positive rate” and the “false positive rate” are derived from the values on the confusion matrix

The confusion matrix plot of the results of training. 
The accuracy for the normal points is very good, but the accuracy for the anomaly points is ok. 
There is still further room for improvement looking at these results, as you have not tuned the hyperparameters of the model yet, 
but it already does ok in detecting anomalies. 
The goal now is to keep the accuracy for the normal points as high as possible, 
or at a high enough level that’s acceptable, 
while raising the accuracy for the anomaly points as high as possible. 
Based on this confusion matrix plot, 
you can now see that the lower AUC score is more accurate at reflecting the true performance of the model. 
You can see that a non-negligible amount of anomalies were falsely classified as normal, 
hence an AUC score of 0.84 is a much better indicator of the model’s performance than the graph’s apparent score of 0.98

 •True positives are values that the model predicts as positive that actually are positive.
• False negatives are values that the model predicts as negative that actually are positive.
• False positives are values that the model predicts as positive that actually are negative.
• True negatives are values that the model predicts as negative that actually are negative.
"""
conf_matrix = confusion_matrix(y_test, preds)
ax = sns.heatmap(conf_matrix, annot=True,fmt='g') 
ax.invert_xaxis()
ax.invert_yaxis()
plt.ylabel('Actual')
plt.xlabel('Predicted')


#################################### Model Validation ####################################

"""
KFold() 函數的作用是將傳入的數據拆分為 num_folds 個不同的分區。 單個折疊一次用作驗證集，而其餘折疊用於訓練。
"""
anomaly_weights = [1, 5, 10, 15]

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=2020)

logs = []
for f in range(len(anomaly_weights)): 
    fold = 1
    accuracies = []
    auc_scores= []
    for train, test in kfold.split(x_validate, y_validate):
        weight = anomaly_weights[f]
        class_weights= {0:1, 1: weight}
        sk_model = LogisticRegression(
                    random_state=None, max_iter=400,
                    solver='newton-cg',
                    class_weight=class_weights).fit(x_validate[train],y_validate[train])

for h in range(40): 
    print('-', end="") 
    print(f"\nfold {fold}\nAnomaly Weight: {weight}")
    eval_acc = sk_model.score(x_validate[test],
    y_validate[test])
    preds = sk_model.predict(x_validate[test])
    try:
        auc_score = roc_auc_score(y_validate[test], preds)
    except:
        auc_score = -1
    print("AUC: {}\neval_acc: {}".format(auc_score, eval_acc))
    accuracies.append(eval_acc)
    auc_scores.append(auc_score)
    log = [sk_model, x_validate[test], y_validate[test], preds] 
    logs.append(log)
    fold = fold + 1
print("\nAverages: ")
print("Accuracy: ", np.mean(accuracies))
print("AUC: ", np.mean(auc_scores))
print("Best: ")
print("Accuracy: ", np.max(accuracies))
print("AUC: ", np.max(auc_scores))

#First, you load the correct log in the list of logs. Since the anomaly weight was 10, and the second fold performed the best, you want to look at the twelfth index in the entries in logs.
sk_model, x_val, y_val, preds = logs[11]
roc_plot = plot_roc_curve(sk_model, x_val, y_val, name='Scikit-learn ROC Curve')

conf_matrix = confusion_matrix(y_val, preds)
ax = sns.heatmap(conf_matrix, annot=True,fmt='g') 
ax.invert_xaxis()
ax.invert_yaxis()
plt.ylabel('Actual')
plt.xlabel('Predicted')
