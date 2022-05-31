
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.svm import LinearSVC
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def RandomForestClassification(X_train,Y_train):
    clf=RandomForestClassifier(n_estimators=2000)
    clf.fit(X_train,Y_train)
    return clf
    
def linearSVCclassifier():
    clf=LinearSVC(C=300.0)
    # clf.fit(X_train,Y_train )
    return clf

def SVM_SVC_classifier( ):
    clf=svm.SVC(C=300.0)
    # clf.fit(X_train,Y_train )
    return clf

def AdaBoostClassification(X_train,Y_train):
    clf=AdaBoostClassifier(n_estimators=400)
    clf.fit(X_train,Y_train)
    return clf

def Gradient_Boost_Classifier(X_train,Y_train):
    clf=GradientBoostingClassifier(n_estimators=120, learning_rate=1.0, max_depth=8,random_state=0)
    clf.fit(X_train,Y_train)
    return clf

def MLP():
    clf= MLPClassifier(random_state=0, max_iter=100, hidden_layer_sizes=2)
    # clf.fit(X_train,Y_train)
    return clf

def logistic_regression():
    clf=LogisticRegression()
    # clf.fit(X_train,Y_train)
    return clf

# def votingClassification(X_train,Y_train):
#     forest = RandomForestClassification()
#     svc = linearSVCclassifier()
#     # svm = SVM_SVC_classifier( )
#     adaboost = AdaBoostClassification()
#     gradient = Gradient_Boost_Classifier()
#     mlp = MLP()
#     # regression = logistic_regression()
#     clf = VotingClassifier(estimators=[('gradient',gradient)
#     # ,('svm',svm),('regression',regression)
#     , ('forest',forest), ('svc',svc), ('adaboost',adaboost),('mlp',mlp)], voting='hard')
#     clf.fit(X_train,Y_train)
#     return clf






