import numpy as np
import pandas as pd
import math
import random
class internalNode:
    def __init__ (self, left, right, SplitAtt, SplitValue):
        self.left = left
        self.right = right
        self.SplitAtt = SplitAtt
        self.SplitValue = SplitValue

class externalNode:
    def __init__(self, size):
        self.size = size

# X : subsample_input_data
# e : current_tree_height
# l : height_limit_of_a_tree
def isolationTree(X, e, l):
    if e >= l or len(X) <= 1:
        return externalNode(len(X))
    else:
        while True:
            q = random.randrange(0, len(X[0]), 1)
            max_value, min_value = Imaxmin(X, q)
            if max_value != min_value:
                break
        p = np.random.uniform(low = min_value, high = max_value)                # chọn ngẫu nhiên 1 điểm chia p từ max & min values của thuộc tính q trong X
        X_left, X_right = left_right(X,q,p)
        return internalNode(
            isolationTree(X_left, e+1, l),
            isolationTree(X_right, e+1, l),
            q,
            p
        )

def Imaxmin(X, q):
    val = []
    for r in X:
        val.append(r[q])
    return max(val), min(val)

def left_right(X, q, p):
    X_left = []
    X_right = []
    for r in X:
        if(r[q] <= p):
            X_left.append(r)
        else:
            X_right.append(r)
    return X_left, X_right


# data : the whole data
# t    : number of trees
# psy  : sub-sampling size
def isolationForest(data, t, psy):
    Forest=[]
    height_limit_of_a_tree = np.ceil(np.log2(psy))
    l = height_limit_of_a_tree
    
    for __ in range(t):
        sub_sampling_data = random.sample(list(data),psy)
        X = sub_sampling_data
        Forest.append(isolationTree(X, 0, l))
    return Forest

# compute the harmonic-number H(i)
def H(n):
    harmonic = 1
    for i in range(2,n+1):
        harmonic += 1/i
    return harmonic

# compute the average-path-length of unsuccessful seach in BST : c(n)
def compute_the_average_path_length(n):
    return 2.0*H(n-1) - (2.0*(n-1)/n)
c256 = compute_the_average_path_length(256)
print(c256)


# x : an instance
# T : an isolation tree
# e : current path length
def PathLength(x, T, e):
    if isinstance(T, externalNode):
        return e + compute_the_average_path_length(T.size)
    
    a = T.SplitAtt
    if x[a] <= T.SplitValue:
        return PathLength(x, T.left, e+1)
    else:
        return PathLength(x, T.right, e+1)


def Evaluation_Stage(data):
    F    = []
    F    = isolationForest(data, 100, 256)
    anomaly_score = [0 for i in data]
    Outliner = []
    PointO = []
    for i in range(len(data)):
        h = []
        for T in F:
            h.append(PathLength(data[i], T, 0))
        h = np.array(h)
        mean_of_h = np.average(h)
        # compute the anomaly-score s(x,n)
        anomaly_score[i] = math.pow(2, -(mean_of_h/c256))
        if(anomaly_score[i]>0.6):
            Outliner.append(1)
            PointO.append(data[i])
        else:
            Outliner.append(0)
    return Outliner, PointO

data = pd.read_csv('creditcard_data.csv')
data = data.iloc[:].values


outlier, PointO = Evaluation_Stage(data)
print(PointO)

# test github
