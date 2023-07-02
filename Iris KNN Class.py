from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()

features=iris.data
labels=iris.target

# print(iris.DESCR)
# print(features[0],labels[0])

clf=KNeighborsClassifier()

clf.fit(features,labels)

pred=clf.predict([[2.3,0.5,0.5,1.3]])

if(pred[0]==0):
    print("Setosa")
elif pred[0]==1:
    print("Versicolour")
else:
    print("Virginica")
