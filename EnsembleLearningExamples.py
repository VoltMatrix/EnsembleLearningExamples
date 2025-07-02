from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost


#The following code creates and train a voting classifier in SKlearn

log_clf= LogisticRegression()
rnd_clf= RandomForestClassifier()
svm_clf=SVC()

voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
voting='hard')
voting_clf.fit(X_train, y_train)

#Looking at the classifier's accuracy on the test set:
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#Bagging and pasting in sklearn 
bag_clf= BaggingClassifier(
    DecisionTreeClassifier(),n_estimators=500,
    max_samples=100,bootstrap=True,n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred= bag_clf.predict(X_test)


#Out of Bag Evaluation
bag_clf= BaggingClassifier(
    DecisionTreeClassifier(),n_estimators=500,
    bootstrap=True, n_jobs=-1,oob_score=True,
)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_#This will tell the oob score

y_pred= bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)

bag_clf.oob_decision_function_

#To train a Random Forest Classifier
rnd_clf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)


bag_clf= BaggingClassifier(
    DecisionTreeClassifier(splitter="random",max_leaf_nodes==16),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
)

#Extra-Trees
# When you are growing a tree in a Random Forest, at each node only a random subset
#of the features is considered for splitting (as discussed earlier). It is possible to make
#trees even more random by also using random thresholds for each feature rather than
#searching for the best possible thresholds (like regular Decision Trees do).



# the following code
 #trains a RandomForestClassifier on the iris dataset (introduced in Chapter 4) and
 #outputs each feature’s importance. It seems that the most important features are the
 #petal length (44%) and width (42%), while sepal length and width are rather unim
#portant in comparison (11% and 2%, respectively)



iris= load_iris()
rnd_clf= RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name,score in zip(iris["feature_names"],rnd_clf.feature_importances_):
    print(name,score)


ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),n_estimators=200,
    algorithm="SAMME.R",learning_rate=0.5
)
ada_clf.fit(X_train, y_train)

#Gradient Boosting
tree_reg1= DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X,y)

#Now we train a second DecisionTreeRegressor on residual errors made by first predictor
y2= y-tree_reg1.predict(X)
tree_reg2= DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X,y2)

#Training a 3rd predictor
y3= y2 - tree_reg2.predict(X)
tree_reg3= DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X,y3)

#Now we have an ensemble containing 3 trees
#Make predictions on a new instance simply by adding up the predictions of all the trees
y_pred= sum(tree.predict(X_new) for tree in (tree_reg1 ,tree_reg2 ,tree_reg3))


#To train GBRT to use Scikit-Learn's GradientBoostingRegressorClass
gbrt= GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)

#The following code trains a GBRT ensemble with
#120 trees, then measures the validation error at each stage of training to find the opti
#mal number of trees, and finally trains another GBRT ensemble using the optimal
#number of trees:


X_train, X_val, y_train, y_value= train_test_split(X,y)
gbrt= GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train,y_train)

errors= [mean_squared_error(y_val, y_pred)
         for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(
    max_depth=2, n_estimators=bst_n_estimators)

gbrt_best.fit(X_train, y_train)



#implementing early stopping
gbrt= GradientBoostingRegressor(max_depth=2, warm_start=True)

min_val_error= float("inf")
error_going_up= 0
for n_estimators in range(1,120):
    gbrt.n_estimators= n_estimators
    gbrt.fit(X_train, y_train)
    y_pred= gbrt.predict(X_val)
    val_error= mean_squared_error(y_val, y_pred)
    if val_error< min_val_error:
        min_val_error=val_error
        error_going_up=0
    else:
        error_going_up +=1
        if error_going_up == 5:
            break    #early stopping


#Using XGBoost
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)


xgb_reg.fit(X_train,y_train,eval_set=[(X_val, y_val)],early_stopping_rounds=2)
y_pred= xgb_reg.predict(X_val)






#STACKING

























