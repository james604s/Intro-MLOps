
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np 
import subprocess
import json

df = pd.read_csv("dataset/creditcard.csv")
def query(input_json):
    proc = subprocess.run(["curl", "-X", "POST", "-H",
    "Content-Type:application/json; format=pandas-split",
                        "--data", input_json,
                        "http://35.229.188.80:5000/invocations"],
                        stdout=subprocess.PIPE, encoding='utf-8')
    output = proc.stdout
    print(2,output)
    preds = json.loads(output)
    print(3,preds)
    return preds

#querying the model with the first 80 rows of your data frame
input_json = df.iloc[:80].drop(["Time", "Class"],axis=1).to_json(orient="split")
pd.DataFrame(query(input_json)).T


normal = df[df.Class == 0].sample(frac=0.5, random_state=2020).reset_index(drop=True)
anomaly = df[df.Class == 1]
normal_train, normal_test = train_test_split(normal, test_size= 0.2, random_state = 2020)
anomaly_train, anomaly_test = train_test_split(anomaly,test_size = 0.2, random_state = 2020)

scaler = StandardScaler()
scaler.fit(pd.concat((normal, anomaly)).drop(["Time", "Class"],axis=1))
test = pd.concat((normal.iloc[:1900], anomaly.iloc[:100]))
true = test.Class
test = scaler.transform(test.drop(["Time", "Class"], axis=1))

preds = []
batch_size = 80
for f in range(25):
    print(f"Batch {f}", end=" - ")
    sample = pd.DataFrame(test[f*batch_size:(f+1)*batch_size]).to_json(orient="split")
    output = query(sample)
    resp = pd.DataFrame([output])
    preds = np.concatenate((preds, resp.values[0]))
    print("Completed")
eval_acc = accuracy_score(true, preds)
eval_auc = roc_auc_score(true, preds)
print("Eval Acc", eval_acc)
print("Eval AUC", eval_auc)