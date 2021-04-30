import pandas as pd
import math
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, precision_recall_curve, roc_auc_score, confusion_matrix

class DataSets():
    def __init__(self, XTrainEng, XTestEng, YTrainEng, YTestEng, XTrainMath8, XTestMath8, YTrainMath8, YTestMath8, XEng, YEng, XMaths8, YMaths8):
        self.XTrainEng = XTrainEng
        self.XTestEng = XTestEng
        self.YTrainEng = YTrainEng
        self.YTestEng = YTestEng
        self.XTrainMath8 = XTrainMath8
        self.XTestMath8 = XTestMath8
        self.YTrainMath8 = YTrainMath8
        self.YTestMath8 = YTestMath8
        self.XEng = XEng
        self.YEng = YEng
        self.XMaths8 = XMaths8
        self.YMaths8 = YMaths8

    def XTrainEng(self):
        return self.XTrainEng
    def XTestEng(self):
        return self.XTestEng
    def YTrainEng(self):
        return self.YTrainEng
    def YTestEng(self):
        return self.YTestEng
    def XTrainMath8(self):
        return self.XTrainMath8
    def XTestMath8(self):
        return self.XTestMath8
    def YTrainMath8(self):
        return self.YTrainMath8
    def YTestMath8(self):
        return self.YTestMath8
    def XEng(self):
        return self.XEng
    def YEng(self):
        return self.YEng
    def XMaths8(self):
        return self.XMaths8
    def YMaths8(self):
        return self.YMaths8

def loadData():

    ##SMOTE using most samples possible, 9 for eng, 2 for maths
    sm = SMOTE(random_state=1, k_neighbors=5)
    ##read in data to pd dataframe
    data = pd.read_csv('AllResultsPreProcessed.csv')

    ##remove grade 9 rows as only 2 - not enough for SMOTE
    mNines = data.loc[data['MathsAct'] == 9].index
    dataMaths = data.drop(mNines)
    ##seperate X and Y for 8 classes maths
    Xmath8 = dataMaths.iloc[:,[4,5,6,7,8,11,14]].values
    Ymath8 = dataMaths.iloc[:,15].values

    XSmoteMath8, YSmoteMath8 = sm.fit_sample(Xmath8, Ymath8)

    ##select features; HML,Re, Wr, Ma, Att8Est, EngEst, MathEst
    X = data.iloc[:,[4,5,6,7,8,11,14]].values
    ## select target classes for eng and maths
    engY = data.iloc[:,12].values
    mathY = data.iloc[:,15].values

    ##get teachers predicted grades for base data
    engTPreds = data.iloc[:,11].values
    mathsTPreds = data.iloc[:,14].values

    ##convert teachers predictions t discrete values
    for x in range(len(engTPreds)):
        engTPreds[x] = math.floor(engTPreds[x])

    for x in range(len(mathsTPreds)):
        mathsTPreds[x] = math.floor(mathsTPreds[x])

    print("Teacher's prediction's metrics")
    print("English")
    print("Precision: ", precision_score(engY, engTPreds, average='weighted'))
    print("recall: ", recall_score(engY, engTPreds, average='weighted'))
    print("F1: ", f1_score(engY, engTPreds, average='weighted'))
    conMatrixTPE = confusion_matrix(engY, engTPreds)
    print(conMatrixTPE)

    print("Maths")
    print(len(mathY), " ", len(mathsTPreds))
    print("Precision: ", precision_score(mathY, mathsTPreds, average='weighted'))
    print("recall: ", recall_score(mathY, mathsTPreds, average='weighted'))
    print("F1: ", f1_score(mathY, mathsTPreds, average='weighted'))
    conMatrixTPM = confusion_matrix(mathY, mathsTPreds)
    print(conMatrixTPM)

    cme = pd.DataFrame(conMatrixTPE, ['a1','a2','a3','a4','a5','a6','a7','a8','a9'],['p1','p2','p3','p4','p5','p6','p7','p8','p9'])

    cmm = pd.DataFrame(conMatrixTPM, ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'],
                   ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9'])

    fig = plt.figure(figsize=(18,9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('English')
    ax2.title.set_text('Maths')
    sn.set(font_scale=1)
    sn.heatmap(cme, annot=True, annot_kws={'size': 16}, fmt='g', ax=ax1)
    sn.heatmap(cmm, annot=True, annot_kws={'size': 16}, fmt='g', ax=ax2)

    plt.show()

    ##use SMOTE to oversample minority classes
    XSmoteEng, YSmoteEng = sm.fit_sample(X, engY)

    ##Split datasets for training and test 80/20
    XTrainEng, XTestEng, YTrainEng, YTestEng = train_test_split(XSmoteEng, YSmoteEng, test_size=0.2, random_state=1)
    XTrainMath8, XTestMath8, YTrainMath8, YTestMath8 = train_test_split(XSmoteMath8, YSmoteMath8, test_size=0.2, random_state=1)

    return DataSets(XTrainEng, XTestEng, YTrainEng, YTestEng, XTrainMath8, XTestMath8, YTrainMath8, YTestMath8,
                 XSmoteEng,YSmoteEng, XSmoteMath8, YSmoteMath8)
