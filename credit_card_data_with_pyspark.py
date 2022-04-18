import pyspark 
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as LogisticRegressionPySpark
import pyspark.sql.functions as F
import os
import seaborn as sns
import sklearn 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib 
import matplotlib.pyplot as plt

os.environ["SPARK_LOCAL_IP"]='127.0.0.1'
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.sparkContext._conf.getAll()
# spark.stop()
print("pyspark: {}".format(pyspark.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("seaborn: {}".format(sns.__version__))
print("sklearn: {}".format(sklearn.__version__))

data_path = 'dataset/creditcard.csv'
df = spark.read.csv(data_path, header = True, inferSchema = True) 
labelColumn = "Class"
columns = df.columns
numericCols = columns
numericCols.remove(labelColumn)
print(numericCols)

df.show(2)

df.toPandas().head()


stages = []
assemblerInputs = numericCols
assembler = VectorAssembler(inputCols=assemblerInputs,
outputCol="features")
stages += [assembler]
dfFeatures = df.select(F.col(labelColumn).alias('label'),*numericCols )

normal = dfFeatures.filter("Class == 0").sample(withReplacement=False, fraction=0.5, seed=2020) 
anomaly = dfFeatures.filter("Class == 1")

normal_train, normal_test = normal.randomSplit([0.8, 0.2],seed = 2020)
anomaly_train, anomaly_test = anomaly.randomSplit([0.8, 0.2],seed = 2020)

train = normal_train.union(anomaly_train)
test = normal_test.union(anomaly_test)

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(dfFeatures)
train = pipelineModel.transform(train)
test = pipelineModel.transform(test)
selectedCols = ['label', 'features'] + numericCols
train = train.select(selectedCols)
test = test.select(selectedCols)
print("Training Dataset Count: ", train.count())
print("Test Dataset Count: ", test.count())
