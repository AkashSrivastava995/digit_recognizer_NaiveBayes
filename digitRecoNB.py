import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

mat_train=pd.read_csv("train.csv")
mat_train=mat_train.as_matrix()
label_train=mat_train[:,0]
print(np.shape(label_train))
features_train=mat_train[0:,1:]
print(np.shape(features_train))

mat_test=pd.read_csv("test.csv")
mat_test=mat_test.as_matrix()
label_test=mat_test[:,0]
print(np.shape(label_test))
features_test=mat_test[0:,0:]
print(np.shape(features_test))

clf=GaussianNB()
clf.fit(features_train,label_train)
dat=clf.predict(features_test)
print(dat)
print(np.shape(dat))
print("Writing to File......")
df=pd.DataFrame(data=dat.astype(float))
df.to_csv("output.csv",sep=',',header=False,float_format='%.2f',index=False)

print("Accuracy=")
print(clf.score(features_test,label_test))
