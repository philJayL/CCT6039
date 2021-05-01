import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, precision_recall_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from yellowbrick.model_selection import validation_curve
from sklearn.model_selection import train_test_split

import data as data
##load dataset from data.py
dataset = data.loadData()

###Validation Curve and Exhaustive Grid search adapted from
#  'Optimizing Hyperparameters in Random Forest Classification' R. Meinert
# https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6
# accessed 13-4-2021

##parameter ranges to find best parameters
estimatorsRange = np.arange(1,100,2)
maxFeaturesRange = [2,3,4,5,6,7]
minSplitRange = [2,3,4,5,6]
minLeafRange = [2,3,4,5,6]

##print curves for individual parameters
print(validation_curve(RandomForestClassifier(),dataset.XEng, dataset.YEng,
                       param_name='n_estimators', param_range=estimatorsRange, scoring="accuracy"))
print(validation_curve(RandomForestClassifier(),dataset.XEng, dataset.YEng,
                       param_name='min_samples_split', param_range=minSplitRange, scoring="accuracy"))
print(validation_curve(RandomForestClassifier(),dataset.XEng, dataset.YEng,
                       param_name='min_samples_leaf', param_range=minLeafRange, scoring="accuracy"))
print(validation_curve(RandomForestClassifier(),dataset.XEng, dataset.YEng,
                       param_name='max_features', param_range=maxFeaturesRange, scoring="accuracy"))

print(validation_curve(RandomForestClassifier(),dataset.XMaths8,dataset.YMaths8,
                       param_name='n_estimators', param_range=estimatorsRange, scoring="accuracy"))
print(validation_curve(RandomForestClassifier(),dataset.XMaths8,dataset.YMaths8,
                       param_name='min_samples_split', param_range=minSplitRange,scoring="accuracy"))
print(validation_curve(RandomForestClassifier(),dataset.XMaths8,dataset.YMaths8,
                       param_name='min_samples_leaf', param_range=minLeafRange, scoring="accuracy"))
print(validation_curve(RandomForestClassifier(),dataset.XMaths8,dataset.YMaths8,
                       param_name='max_features', param_range=maxFeaturesRange, scoring="accuracy"))

##Exhaustive grid search to find best parameters
print("Exhaustive Grid Search...")

hyperF = dict(n_estimators = estimatorsRange, max_features = maxFeaturesRange,
              min_samples_split = minSplitRange, min_samples_leaf = minLeafRange)
gridF = GridSearchCV(RandomForestClassifier(), hyperF, cv=3, verbose = 1, n_jobs = -1)

bestFMaths = gridF.fit(dataset.XMaths8, dataset.YMaths8)
print("Best Params Maths", bestFMaths.best_params_)

bestFEng = gridF.fit(dataset.XEng, dataset.YEng)
print("Best Params English", bestFEng.best_params_)

##best params Exhasutive
RFClassifierEng = RandomForestClassifier(n_estimators=89, max_features=3, min_samples_leaf=2,
                                         min_samples_split=2, n_jobs=-1, random_state=1)

##exhaustive params
RFClassifierMath8 = RandomForestClassifier(n_estimators=81, max_features=5, min_samples_leaf=2,
                                           min_samples_split=3, n_jobs=-1, random_state=1)

##list of confusion matrix to average from cross val
conMListEng = []
conMListMaths = []
##list of performance metrics per class
engPrecPerClass = []
engRecPerClass = []
engF1PerClass = []
##list of performance metrics per class
mathsPrecPerClass = []
mathsRecPerClass = []
mathsF1PerClass = []

##cross validation tests
for x in range(0,5):
    ##split dataset
    xTrainE, XTestE, YTrainE, YTestE = train_test_split(dataset.XEng, dataset.YEng,
                                                        test_size=0.2, random_state= x+1)
    ##fit and test english model
    RFClassifierEng.fit(xTrainE,YTrainE)
    engPreds = RFClassifierEng.predict(XTestE)
    conMListEng.append(confusion_matrix(YTestE, engPreds))
    ##collect precision, recall and f1 for each fold
    engPrecPerClass.append(precision_score(YTestE, engPreds, average=None))
    engRecPerClass.append(recall_score(YTestE, engPreds, average=None))
    engF1PerClass.append(f1_score(YTestE, engPreds, average=None))

    ##split dataset
    xTrainM, XTestM, YTrainM, YTestM = train_test_split(dataset.XMaths8, dataset.YMaths8,
                                                        test_size=0.2, random_state= x+1)
    ##fit and test maths model
    RFClassifierMath8.fit(xTrainM, YTrainM)
    mathsPreds = RFClassifierMath8.predict(XTestM)
    conMListMaths.append(confusion_matrix(YTestM, mathsPreds))
    ##collect precision, recall and f1 for each fold
    mathsPrecPerClass.append(precision_score(YTestM, mathsPreds, average=None))
    mathsRecPerClass.append(recall_score(YTestM, mathsPreds, average=None))
    mathsF1PerClass.append(f1_score(YTestM, mathsPreds, average=None))

##collect performance measures per class
classPrecisionEnglish = [0] * 9
classRecallEnglish = [0] * 9
classF1English = [0] * 9

##calculate average metrics for each class
for i in range(0,9):
    ##average per class performance across folds
    for fold in engPrecPerClass:
        classPrecisionEnglish[i] = classPrecisionEnglish[i] + fold[i]
    for fold in engRecPerClass:
        classRecallEnglish[i] = classRecallEnglish[i] + fold[i]
    for fold in engF1PerClass:
        classF1English[i] = classF1English[i] + fold[i]

    classPrecisionEnglish[i] = classPrecisionEnglish[i]/5
    classRecallEnglish[i] = classRecallEnglish[i] / 5
    classF1English[i] = classF1English[i] / 5

print("English performance per class")
print("Precision per Class", classPrecisionEnglish, "average: ", np.mean(classPrecisionEnglish))
print("Recall per Class", classRecallEnglish, "average: ", np.mean(classRecallEnglish))
print("F1 per Class", classF1English, "average: ", np.mean(classF1English))

classPrecisionMaths = [0] * 8
classRecallMaths = [0] * 8
classF1Maths = [0] * 8

##calculate average metrics for each class
for i in range(0,8):
    ##average per class performance across folds
    for fold in mathsPrecPerClass:
        classPrecisionMaths[i] = classPrecisionMaths[i] + fold[i]
    for fold in mathsRecPerClass:
        classRecallMaths[i] = classRecallMaths[i] + fold[i]
    for fold in mathsF1PerClass:
        classF1Maths[i] = classF1Maths[i] + fold[i]

    classPrecisionMaths[i] = classPrecisionMaths[i]/5
    classRecallMaths[i] = classRecallMaths[i] / 5
    classF1Maths[i] = classF1Maths[i] / 5

print("Maths performance per class")
print("Precision per Class: ", classPrecisionMaths, "average: ", np.mean(classPrecisionMaths))
print("Recall per Class: ", classRecallMaths, "average: ", np.mean(classRecallMaths) )
print("F1 per Class", classF1Maths, "average: ", np.mean(classF1Maths))

##average confusion matrixs across folds
meanCME = np.mean(conMListEng, axis=0)
meanCMM = np.mean(conMListMaths, axis=0)

cme = pd.DataFrame(meanCME, ['a1','a2','a3','a4','a5','a6','a7','a8','a9'],
                   ['p1','p2','p3','p4','p5','p6','p7','p8','p9'])
cmm = pd.DataFrame(meanCMM, ['a1','a2','a3','a4','a5','a6','a7','a8'],
                   ['p1','p2','p3','p4','p5','p6','p7','p8'])
##plot confusion matrixs
fig = plt.figure(figsize=(18, 9))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.title.set_text('English')
ax2.title.set_text('Maths')
sn.set(font_scale=1)
sn.heatmap(cme, annot=True, annot_kws={'size': 16}, fmt='g', ax=ax1)
sn.heatmap(cmm, annot=True, annot_kws={'size': 16}, fmt='g', ax=ax2)

plt.show()