import numpy as np
from sklearn import linear_model
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 0, 0])
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X, Y)
clf.partial_fit(X, Y)

print(clf.predict(np.array([[-1,-2],[2,2]])))

a = [[1,1], [12,2], [3,4], [5,6]]
b = [1,3]
print(a[b])