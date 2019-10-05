# importing Libraries pandas, numpy and sklearn
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

#Reading Training data
mat_train=pd.read_csv("train.csv")
# Converting to Matrix
mat_train=mat_train.as_matrix()
label_train=mat_train[:,0]
print(np.shape(label_train))
features_train=mat_train[0:,1:]
print(np.shape(features_train))

# Reading Testing Data
mat_test=pd.read_csv("test.csv")
# Converting to Matrix
mat_test=mat_test.as_matrix()
label_test=mat_test[:,0]
print(np.shape(label_test))
features_test=mat_test[0:,0:]
print(np.shape(features_test))

# Creating a instants of Gaussian Naive Bayes Classifier
clf=GaussianNB()
# Training the Classifier on Train Data 
clf.fit(features_train,label_train)
# predicting the unknown data with Classifier using `predict()`
dat=clf.predict(features_test)

# printing the results
print(dat)
print(np.shape(dat))
# Writing results to file named `Output.csv`
print("Writing to File......")
df=pd.DataFrame(data=dat.astype(float))
df.to_csv("output.csv",sep=',',header=False,float_format='%.2f',index=False)

#printing Accuracy of classifier
print("Accuracy=")
print(clf.score(features_test,label_test))
