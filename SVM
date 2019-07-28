import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC
from sklearn import metrics


data = pd.read_csv("File_Location/File_Name.csv")


X = data.drop('Token', axis=1)  
y = data['Token']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

  
svclassifier = SVC(kernel='rbf', gamma=1.5, C=21)  
svclassifier.fit(X_train, y_train)

pred_train = svclassifier.predict(X_train)
print('\nPrediction accuracy for the training & test with preprocessing: ')
print('{:.2%}'.format(metrics.accuracy_score(y_train, pred_train)))
pred_test = svclassifier.predict(X_test)
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))
print("Predicted Classes: ",pred_test)


LABELS = ['WALKING', 'WALKING UPSTAIRS', 'WALKING DOWNSTAIRS', 'SITTING', 'STANDING',
          'LYING','USING TOILET', 'JOGGING', 'WRITING','TYPING']

import seaborn as sns
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, pred_test)

plt.figure(figsize=(4, 4))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap="YlGnBu");
plt.ylabel('ACTUAL LABELS')
plt.xlabel('PREDICTED LABELS')
plt.show();
