import pandas as pd 
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np 
import subprocess
import json

#"runs:/f23c101d414e43be877210a6f954a1cd/log_reg_model"
# mlflow models serve --model-uri runs:/f23c101d414e43be877210a6f954a1cd/
# f23c101d414e43be877210a6f954a1cd -p 1235


####### Querying the Model #######
data_path = "dataset/creditcard.csv"
df = pd.read_csv(data_path)

input_json = df.iloc[:80].drop(["Time", "Class"],axis=1).to_json(orient="split")

# here is the code to send data to the model and receive predictions back
proc = subprocess.run(["curl",  "-X", "POST", "-H", "Content-Type:application/json; format=pandas-split", "--data", input_json, "http://127.0.0.1:1235/invocations"],
stdout=subprocess.PIPE, encoding='utf-8')
output = proc.stdout
df2 = pd.DataFrame([json.loads(output)])
df2

#curl -X POST -H "Content-Type:application/json; format=pandas-split" –data "CONTENT_OF_INPUT_JSON" "http://127.0.0.1:1235/invocations"

####### Querying without scaling #######

y_true = df.iloc[:80].Class
df2 = df2.T
eval_acc = accuracy_score(y_true, df2)
y_true.iloc[-1] = 1
eval_auc = roc_auc_score(y_true, df2)
print("Eval Acc", eval_acc)
print("Eval AUC", eval_auc)

####### Querying with scaling #######
normal = df[df.Class == 0].sample(frac=0.5, random_state=2020). reset_index(drop=True)
anomaly = df[df.Class == 1]
normal_train, normal_test = train_test_split(normal,test_size = 0.2, random_state = 2020)
anomaly_train, anomaly_test = train_test_split(anomaly, test_size = 0.2, random_state = 2020)
scaler = StandardScaler()
scaler.fit(pd.concat((normal, anomaly)).drop(["Time", "Class"],axis=1))

# fit the scaler, let’s transform your data selection
scaled_selection = scaler.transform(df.iloc[:80].drop(["Time", "Class"], axis=1))
input_json = pd.DataFrame(scaled_selection).to_json(orient="split")

proc = subprocess.run(["curl", "-X", "POST", "-H", "Content-Type:application/json; format=pandas-split", "--data", input_json, "http://127.0.0.1:1235/invocations"], stdout=subprocess.PIPE, encoding='utf-8')
output = proc.stdout
preds = pd.DataFrame([json.loads(output)])
preds


y_true = df.iloc[:80].Class
preds = preds.T
eval_acc = accuracy_score(y_true, preds)
y_true.iloc[-1] = 1
eval_auc = roc_auc_score(y_true, preds)
print("Eval Acc", eval_acc)
print("Eval AUC", eval_auc)

#######Batch Querying  #######
#The results of querying the model with the first 8,000 samples in the data frame. Notice that the AUC score is far better samples
test = df.iloc[:8000]
true = test.Class
test = scaler.transform(test.drop(["Time", "Class"], axis=1))
preds = []
batch_size = 80
for f in range(100):
    sample = pd.DataFrame(test[f*batch_size:(f+1)*batch_size]). to_json(orient="split")
    proc = subprocess.run(["curl",  "-X", "POST", "-H", "Content-Type:application/json; format=pandas-split", "--data", sample, "http://127.0.0.1:1235/ invocations"],
                          stdout=subprocess.PIPE,
                          encoding='utf-8')
    output = proc.stdout
    resp = pd.DataFrame([json.loads(output)])
    preds = np.concatenate((preds, resp.values[0]))
eval_acc = accuracy_score(true, preds)
eval_auc = roc_auc_score(true, preds)
print("Eval Acc", eval_acc)
print("Eval AUC", eval_auc)

conf_matrix = confusion_matrix(true, preds)
ax = sns.heatmap(conf_matrix, annot=True,fmt='g') 
ax.invert_xaxis()
ax.invert_yaxis()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix")

"""

MLFlow is an API that can help you integrate MLOps principles into your existing code base, supporting a wide variety of popular frameworks.
In this chapter, we covered how you can use MLFlow to log metrics, parameters, graphs, and the models themselves. Additionally, you learned how to load the logged model and make use of its functionality. As for frameworks, we covered how you can apply MLFlow to your experiments
in scikit-learn, TensorFlow 2.0/Keras, PyTorch, and PySpark, and we also looked at how you can take one of these models, deploy it locally, and make predictions with your model.
In the next chapter, we will look at how you can take your MLFlow models and use MLFlow functionality to help deploy them to Amazon SageMaker. Furthermore, we will also look at how you can make predictions using your deployed model.
"""