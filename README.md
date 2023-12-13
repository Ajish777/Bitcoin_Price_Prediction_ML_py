
# Bitcoin Price Prediction using Machine Learning in Python

# Importing Libraries

* Pandas – This library helps to load the data frame in a 2D array format and has multiple functions to perform analysis tasks in one go.
* Numpy – Numpy arrays are very fast and can perform large computations in a very short time.
* Matplotlib/Seaborn – This library is used to draw visualizations.
* Sklearn – This module contains multiple libraries having pre-implemented functions to perform tasks from data preprocessing to model development and evaluation.
* XGBoost – This contains the eXtreme Gradient Boosting machine learning algorithm which is one of the algorithms which helps us to achieve high accuracy on predictions.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
```

# Importing Dataset
The dataset used here to perform the analysis and build a predictive model is Bitcoin Price data. I have use OHLC(‘Open’, ‘High’, ‘Low’, ‘Close’) data from 17th July 2014 to 29th December 2022 which is for 8 years for the Bitcoin price.

you can find and use datasets from kaggle.

```
df = pd.read_csv('bitcoin.csv')
df.head()
```
* Output:

![Capture1](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/8bada364-d390-4d13-a5b5-f052c05555e0)

```
df.shape
```
* Output:
```
(2904, 7)
```
From this, we got to know that there are 2904 rows of data available and for each row, we have 7 different features or columns.

```
df.describe()
```
* Output:
![Capture2](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/5ba2b74d-87ec-454a-aa93-ec38c2238e41)

```
df.info()
```
* Output:
![Capture3](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/c4fdd317-9447-45b6-bdc8-def6f42eeba7)

# Exploratory Data Analysis
EDA is an approach to analyzing the data using visual techniques. It is used to discover trends, and patterns, or to check assumptions with the help of statistical summaries and graphical representations. 

While performing the EDA of the Bitcoin Price data we will analyze how prices of the cryptocurrency have moved over the period of time and how the end of the quarters affects the prices of the currency.

```
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
```
* Output:
![Capture4](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/170f591c-d0a4-4125-ae00-a7c5848825fb)

The prices of the Bitcoin stocks are showing an upward trend as depicted by the plot of the closing price of the stocks.

```
df[df['Close'] == df['Adj Close']].shape, df.shape
```

* Output:
```
((2904, 7), (2904, 7))
```

From here we can conclude that all the rows of columns ‘Close’ and ‘Adj Close’ have the same data. So, having redundant data in the dataset is not going to help so, we’ll drop this column before further analysis.

```
df = df.drop(['Adj Close'], axis=1)
```
* Output:

Now let’s draw the distribution plot for the continuous features given in the dataset but before moving further let’s check for the null values if any are present in the data frame.

```
df.isnull().sum()
```
* Output:
![Capture5](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/b6cd906b-6634-4277-95ae-c028b6f743cc)

This implies that there are no null values in the data set provided.

```
features = ['Open', 'High', 'Low', 'Close']

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,2,i+1)
sb.distplot(df[col])
plt.show()
```

* Output:

![Capture6](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/ede43036-ebe0-4d2a-af23-165adde4d7cf)

```
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,2,i+1)
sb.boxplot(df[col])
plt.show()
```
* Output:
![Capture7](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/34d0deec-14c1-49c7-86e5-17bbf977bbf6)

There are so many outliers in the data which means that the prices of the stock have varied hugely in a very short period of time. Let’s check this with the help of a barplot. 

# Feature Engineering
Feature Engineering helps to derive some valuable features from the existing ones. These extra features sometimes help in increasing the performance of the model significantly and certainly help to gain deeper insights into the data.

```
splitted = df['Date'].str.split('-', expand=True)

df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

df.head()
```
* Output:
![Capture8](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/f2da839b-bf84-4eff-be9a-62fbb15e9a93)

Now we have three more columns namely ‘day’, ‘month’ and ‘year’ all these three have been derived from the ‘Date’ column which was initially provided in the data.

```
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
plt.subplot(2,2,i+1)
data_grouped[col].plot.bar()
plt.show()
```
* Output:
![Capture9](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/ba9515ee-a6fa-43c0-9151-296ea55058b0)

Here we can observe why there are so many outliers in the data as the prices of bitcoin have exploded in the year 2021.

```
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()
```

* Output:
![Capture10](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/c138dd5a-9f5b-4570-96af-16ddffa5bfaf)

```
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
```

* Output:
![Capture11](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/45b5b5f6-c652-4bc5-898c-e9d47072e989)

When we add features to our dataset we have to ensure that there are no highly correlated features as they do not help in the learning process of the algorithm.

```
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
```

* Output:

![Capture12](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/aa7b3f03-bcba-42bf-a1ac-62f612bfdb1f)

From the above heatmap, we can say that there is a high correlation between OHLC which is pretty obvious, and the added features are not highly correlated with each other or previously provided features which means that we are good to go and build our model.

```
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target'] 

scaler = StandardScaler()
features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
```
* Output:

```
(2613, 3) (291, 3)
```

After selecting the features to train the model on we should normalize the data because normalized data leads to stable and fast training of the model. After that whole data has been split into two parts with a 90/10 ratio so, that we can evaluate the performance of our model on unseen data.

# Model Development and Evaluation
Now is the time to train some state-of-the-art machine learning models(Logistic Regression, Support Vector Machine, XGBClassifier), and then based on their performance on the training and validation data we will choose which ML model is serving the purpose at hand better.

For the evaluation metric, we will use the ROC-AUC curve but why this is because instead of predicting the hard probability that is 0 or 1 we would like it to predict soft probabilities that are continuous values between 0 to 1. And with soft probabilities, the ROC-AUC curve is generally used to measure the accuracy of the predictions.

```
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
 
for i in range(3):
  models[i].fit(X_train, Y_train)
 
  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()
```
* Output:

![Capture13](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/6299cc55-851b-4fa1-80e2-c6e84c7a50a0)

Among the three models, we have trained XGBClassifier has the highest performance but it is pruned to overfitting as the difference between the training and the validation accuracy is too high. But in the case of the Logistic Regression, this is not the case.

Now let’s plot a confusion matrix for the validation data.

```
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()
```
* Output:
![Capture14](https://github.com/Ajish777/Bitcoin_Price_Prediction_ML_py/assets/110074935/5e4ae8b5-0aed-47e9-ac59-a7e25455ee36)

# Conclusion:
We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.

