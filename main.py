import numpy as np
import csv
import cPickle
from sklearn.metrics import accuracy_score

from numpy import genfromtxt
train_q = genfromtxt('spambase/spambase_train_q.csv', delimiter=',')
train_a = genfromtxt('spambase/spambase_train_ans.csv', delimiter=',')
test_q = genfromtxt('spambase/spambase_test_q.csv', delimiter=',')
actual_a = genfromtxt('spambase/spambase_test_ans.csv', delimiter=',')
#first 1
test_q1 = [[0, 0.64, 0.64, 0, 0.32, 0, 0, 0, 0, 0, 0, 0.64, 0, 0, 0, 0.32, 0, 1.29, 1.93, 0, 0.96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.778, 0, 0, 3.756, 61, 278]]
#second 1
test_q2 = [[0.21, 0.28, 0.5, 0, 0.14, 0.28, 0.21, 0.07, 0, 0.94, 0.21, 0.79, 0.65, 0.21, 0.14, 0.14, 0.07, 0.28, 3.47, 0, 1.59, 0, 0.43, 0.43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.132, 0, 0.372, 0.18, 0.048, 5.114, 101, 1028]]
#last 0
test_q3 = [[0, 0, 0.65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.65, 0, 0, 0, 0, 0, 4.6, 0, 0.65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.97, 0.65, 0, 0, 0, 0, 0, 0.125, 0, 0, 1.25, 5, 40]]
#second last 0
test_q4 = [[0.96, 0, 0, 0, 0.32, 0, 0, 0, 0, 0, 0, 0.32, 0, 0, 0, 0, 0, 0, 1.93, 0, 0.32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.32, 0, 0.32, 0, 0, 0, 0.057, 0, 0, 0, 0, 1.147, 5, 78]]
#third last 0
test_q5 = [[0.3, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 1.8, 0.3, 0, 0, 0, 0, 0.9, 1.5, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.2, 0, 0, 0.102, 0.718, 0, 0, 0, 0, 1.404, 6, 118]]

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(train_q, train_a)
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)
print(clf.predict(test_q1))
print(clf.predict(test_q2))
print(clf.predict(test_q3))
print(clf.predict(test_q4))
print(clf.predict(test_q5))
predicted_a = clf.predict(test_q)
#0.792079207921
print(accuracy_score(actual_a, predicted_a))
