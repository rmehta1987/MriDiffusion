import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import glob
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from visdom import Visdom
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from itertools import compress
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn import feature_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC

#Global Variables

#Classifier Names
names = ["Linear_SVM", "RBF_SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "AdaBoost",
         "Naive_Bayes"]

    
#Feature Names, a10 represents 10th percentile and so on.    
fnames = ["a10", "a25", "a50", "a75", "aquartile", "amean", "amid", "avar", "akurt", 
"askew",  "b10", "b25", "b50", "b75", "bquartile", "bmean", "bmid", "bvr", "bkurt", 
"bskew", "d10", "d25", "d50", "d75", "dquartile","dmean", "dmid", "dvr", "dkurt", "dskew"]         

f2names = ["a10", "a25", "a50", "a75", "aquartile", "amean", "amid", "avar", "akurt", 
"askew",  "b10", "b25", "b50", "b75", "bquartile", "bmean", "bmid", "bvr", "bkurt", 
"bskew", "d10", "d25", "d50", "d75", "dquartile","dmean", "dmid", "dvr", "dkurt", "dskew",
"diff10", "diff25", "diff50", "diff75", "diffquartile", "diffmean", "diffmid", "diffvar", "diffkurt", "diffskew",
"perf10", "perf25", "perf50", "perf75", "perfquartile", "perfmean", "perfmid", "perfvar", "perfkurt", "perfskew",
"f10", "f25", "f50", "f75", "fquartile", "fmean", "fmid", "fvar", "fkurt", "fskew"]       

def getFiles(file_path, name):

    afiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[0])))
    bfiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[1])))
    dfiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[2])))
    difffiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[3])))
    perffiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[4])))
    ffiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[5])))
    lafiles = []
    lbfiles = []
    ldfiles = []
    ldifffiles = []
    lperffiles = []
    lffiles = []
    print ("obtaining files")
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles,bfiles,dfiles,difffiles,perffiles,ffiles)):
        lafiles.append(np.load(a))
        lbfiles.append(np.load(b))
        ldfiles.append(np.load(d))
        ldifffiles.append(np.load(e))
        lperffiles.append(np.load(f))
        lffiles.append(np.load(g))
    
    return lafiles, lbfiles, ldfiles, ldifffiles, lperffiles, lffiles

def createFeatMat(train, test, afiles, bfiles, dfiles):

    xtrain = np.zeros((len(train),30)) #Feature Matrix [a-features, b-features, d-features]
    ytrain = np.zeros((len(train))) #label matrix
    for i, trainidx in enumerate(train):
        temphist = afiles[trainidx][3]
        xtrain[i][0:10] = temphist
        temphist = bfiles[trainidx][3]
        xtrain[i][10:20] = temphist
        temphist = dfiles[trainidx][3]
        xtrain[i][20:] = temphist
        ytrain[i] = afiles[trainidx][1]

    xtest = []
    ytest = []
    if len(test) > 0:
        xtest = np.zeros((len(test),30)) #Feature Matrix [a-features, b-features, d-features]
        ytest = np.zeros((len(test))) #label matrix
        for i, testidx in enumerate(test):
            temphist = afiles[testidx][3]
            xtest[i][0:10] = testidx
            temphist = bfiles[testidx][3]
            xtest[i][10:20] = temphist
            temphist = dfiles[testidx][3]
            xtest[i][20:] = temphist
            ytest[i] = afiles[testidx][1]
    

    return xtrain, ytrain, xtest, ytest

#def createSmallFeatMat(train, test, indices):


def createFeatMat2(afiles, bfiles, dfiles):
    '''creates a feature matrix from the different set of files
    @param afiles = features of amaps  - arranged as the map, label, patient name, histogram features, histogram in a numpy array
    @param bfiles = features of bmaps
    @param dfiles = features of dmaps'''

    xtrain = np.zeros((len(afiles),len(afiles[0][3])*3)) #Feature Matrix [a-features, b-features, d-features]
    ytrain = np.zeros((len(afiles),),dtype=np.int) #label matrix
    for i, (a,b,d) in enumerate(zip(afiles,bfiles,dfiles)):
        xtrain[i] = np.hstack((a[3], b[3], d[3]))
        ytrain[i] = a[1]

    print ("finished creating Feature and Label Matrix")
    return xtrain, ytrain

#IVIM FEATURE MATRIX
def createFeatMat3(afiles, bfiles, dfiles, diff_files, perf_files, f_files):
    '''creates a feature matrix from the different set of files
    @param afiles = features of amaps  - arranged as the map, label, patient name, histogram features, histogram in a numpy array
    @param bfiles = features of bmaps
    @param dfiles = features of dmaps'''

    xtrain = np.zeros((len(afiles),len(afiles[0][3])*6)) #Feature Matrix [a-features, b-features, d-features]
    ytrain = np.zeros((len(afiles),),dtype=np.int) #label matrix
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles,bfiles,dfiles,diff_files, perf_files, f_files)):
        xtrain[i] = np.hstack((a[3], b[3], d[3], e[3], f[3], g[3]))
        ytrain[i] = a[1]

    print ("finished creating Feature and Label Matrix")
    return xtrain, ytrain


viz = Visdom()
h = .02  # step size in the mesh
classifiers = [
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()]

print ("Getting Files containing features of original maps")
file_path = 'maxminFeatIvim'
name = ['mm_apad_','mm_bpad_','mm_dpad_', 'mm_diffpad_', 'mm_perfpad_', 'mm_fpad_']
afiles, bfiles, dfiles, diff_files, perf_files, f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Crop 1")
file_path = 'maxminAugFeatIvim'
name = ['cropaug1_alpha_','cropaug1_beta_','cropaug1_ddc_', 'cropaug1_diff', 'cropaug1_perf', 'cropaug1_f']
c1afiles, c1bfiles, c1dfiles, c1diff_files, c1perf_files, c1f_files = getFiles(file_path,name)


print ("Getting files of augmented maps -- Crop 2")
file_path = 'maxminAugFeatIvim'
name = ['cropaug2_alpha_','cropaug2_beta_','cropaug2_ddc_', 'cropaug2_diff', 'cropaug2_perf', 'cropaug2_f']
c2afiles, c2bfiles, c2dfiles, c2diff_files, c2perf_files, c2f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Crop 3")
file_path = 'maxminAugFeatIvim'
name = ['cropaug3_alpha_','cropaug3_beta_','cropaug3_ddc_', 'cropaug3_diff', 'cropaug3_perf', 'cropaug3_f']
c3afiles, c3bfiles, c3dfiles, c3diff_files, c3perf_files, c3f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Original Augmented Rotated")
file_path = 'maxminAugFeatIvim'
name = ['ogaug_alpha_','ogaug_beta_','ogaug_ddc_', 'ogaug_diff', 'ogaug_perf', 'ogaug_f']
augafiles, augbfiles, augdfiles, augdiff_files, augperf_files, augf_files = getFiles(file_path,name)



#create a Stratified K-fold n-times
rkf = RepeatedKFold(n_splits=3, n_repeats=2)

# iterate over datasets

#####Visualisation of different plots for classifiers
"""
    Linear_SVM = viz.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='iteration',ylabel='F1_Score',title='Linear'))

    RBF_SVM = viz.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='iteration',ylabel='F1_Score',title='RBF'))

    Gaussian_Process = viz.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='iteration',ylabel='F1_Score',title='GP'))

    Decision_Tree = viz.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='iteration',ylabel='F1_Score',title='DT'))

    Random_Forest = viz.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='iteration',ylabel='F1_Score',title='RF'))

    Neural_Net = viz.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='iteration',ylabel='F1_Score',title='F1_scores')) 

    AdaBoost = viz.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='iteration',ylabel='F1_Score',title='Ada'))

    Naive_Bayes = viz.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='iteration',ylabel='F1_Score',title='NB')) 
"""

#Importance of features obtained from RFE Linear SVM, RBF_SVM, and Random Forest -- 
importances = [None] * 3
for i in range(0,3):
    importances[i] = list()


#standard deviation of importance obtained from random forest
std = list()

#scores of accuracy and f1 for each classifier
accscore = np.empty((len(classifiers),0)).tolist()
fonescore = np.empty((len(classifiers),0)).tolist()

#create Feature matrix for the original and augmented files
print ("Getting feature matrix of original maps")
xtrain, ytrain = createFeatMat2(afiles, bfiles, dfiles)

print ("Getting feature matrix of Crop 1 maps")
c1xtrain, c1ytrain = createFeatMat2(c1afiles, c1bfiles, c1dfiles)
print ("Getting feature matrix of Crop 2 maps")
c2xtrain, c2ytrain = createFeatMat2(c2afiles, c2bfiles, c2dfiles)
print ("Getting feature matrix of Crop 3 maps")
c3xtrain, c3ytrain = createFeatMat2(c3afiles, c3bfiles, c3dfiles)
print ("Getting feature matrix of Augmented orignal maps")
augxtrain, augytrain = createFeatMat2(augafiles, augbfiles, augdfiles)

#full augmented feature matrix
full_aug_xtrain = np.vstack([c1xtrain,c2xtrain,c3xtrain,augxtrain])
full_aug_ytrain = np.vstack([c1ytrain.reshape(-1,1),c2ytrain.reshape(-1,1),c3ytrain.reshape(-1,1),augytrain.reshape(-1,1)])

#train only on original set:


for train, test in rkf.split(xtrain):

    txtrain = xtrain[train]
    tytrain = ytrain[train]
    txtest = xtrain[test]
    tytest = ytrain[test]

   # iterate over classifiers
    for i, (name, clf) in enumerate(zip(names, classifiers)):
        
   
        #creating training matrix 
        if name == "Linear_SVM":
            estimator = RFE(clf, 10, step=1)
            estimator = estimator.fit(txtrain, tytrain)
            ypred = estimator.predict(txtest)
            importances[0].append(estimator.get_support(indices=True))
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
        elif name == "RBF_SVM":
            #rbfestimator = RFE(clf, 5, step=1)
            #rbfestimator = clf.fit(txtrain, tytrain)
            rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
            #importances[1].append(rbfestimator.get_support(indices=True))
            ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
        elif name == 'Random_Forest':
            probsz = clf.fit(txtrain, tytrain)
            timportance = clf.feature_importances_
            importances[2].append(timportance)
            std.append(np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0))
            ypred = clf.predict(txtest)
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
        else:
            probsz = clf.fit(txtrain, tytrain)
            ypred = clf.predict(txtest)
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
            

      
        accscore[i].append(theaccscore)
        fonescore[i].append(thef1score)

#find average of feature importances from random forest
avgimport = np.average(np.array(importances[2]),0)
indices = np.argsort(avgimport)[::-1]
avgstd = np.average(np.array(std),0)


print ("RFE Importance")
#print (importances[1][0])
for idx in importances[0][0]:
    print ("Feature %s has importance in RFE Gaussian"%(fnames[idx]))

print ("Random Forest Importance")

for ti in indices:
    print ("Feature %s has probability %f" %(fnames[ti], avgimport[ti]))

print ("Original Data F1 Scores")

r = len(fonescore) #number of classifiers
c = len(fonescore[0])#number of folds 
X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
Y = np.ones((c,r))
Y = Y*np.array(fonescore).transpose()
f1win = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='F-1 score',title='Testing F1 Score of Classifiers',legend=names))

for i,scores in enumerate(fonescore):
    #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=orgf1score,opts=dict(xlabel='Fold',ylabel='F1',title='{} Original F1 score'.format(names[i])))
    #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=orgf1score,opts=dict(xlabel='Fold',ylabel='F1'),update='append')
    print ("Classifier name and F1 Score: %s : "%(names[i]), end = " ")
    print (scores) 

print ("Original Data Accuracy Scores")

for i,scores in enumerate(accscore):
    #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=accwin, opts=dict(xlabel='Fold',ylabel='F1',title='{} Original Accuracy score'.format(names[i])))
    #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=accwin, opts=dict(xlabel='Fold',ylabel='Accuracy'),update='append')
    #viz.line(X=np.arange(1,len(scores)+1).reshape(-1,1),Y=np.array(scores).reshape(-1,1),name=names[i],win=accwin, opts=dict(xlabel='Fold',ylabel='Accuracy',legend=[names[i]]), update='append')
    print ("Classifier name and Accuracy Score: %s : "%(names[i]), end=" ")
    print (scores) 
    


#Stack Visualizations of each accuracy
r = len(accscore) #number of classifiers
c = len(accscore[0])#number of folds 
X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
Y = np.ones((c,r))
Y = Y*np.array(accscore).transpose()
accwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing Accuracy of Classifiers',legend=names))



#viz.bar(avgimport)



print ("Testing Improvement of Augmented Data")

#Random Shuffle of Augmentations

#Importance of features obtained from RFE Linear SVM, RBF_SVM, and Random Forest -- 
importances = [None] * 3
for i in range(0,3):
    importances[i] = list()
#standard deviation of importance obtained from random forest
std = list()

#scores of accuracy and f1 for each classifier
accscore = np.empty((len(classifiers),0)).tolist()
fonescore = np.empty((len(classifiers),0)).tolist()

randindx = np.random.choice(np.arange(len(full_aug_xtrain)),size=int(len(full_aug_xtrain)*0.5),replace=True)

xtrain = np.vstack((xtrain,full_aug_xtrain[randindx]))
ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain[randindx]))

for train, test in rkf.split(xtrain):

    txtrain = xtrain[train]
    tytrain = ytrain[train]
    txtest = xtrain[test]
    tytest = ytrain[test]

   # iterate over classifiers
    for i, (name, clf) in enumerate(zip(names, classifiers)):
        
   
        #creating training matrix 
        if name == "Linear_SVM":
            estimator = RFE(clf, 10, step=1)
            estimator = estimator.fit(txtrain, tytrain.ravel())
            importances[0].append(estimator.get_support(indices=True))
            ypred = estimator.predict(txtest)
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
        elif name == "RBF_SVM":
            #rbfestimator = RFE(clf, 5, step=1)
            rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
            #importances[1].append(rbfestimator.get_support(indices=True))
            ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)

        elif name == 'Random_Forest':
            probsz = clf.fit(txtrain, tytrain.ravel())
            timportance = clf.feature_importances_
            importances[2].append(timportance)
            std.append(np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0))
            ypred = clf.predict(txtest)
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)

        else:
            probsz = clf.fit(txtrain, tytrain.ravel())
            ypred = clf.predict(txtest)
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)



      
        accscore[i].append(theaccscore)
        fonescore[i].append(thef1score)



#find average of feature importances from random forest
avgimport = np.average(np.array(importances[2]),0)
indices = np.argsort(avgimport)[::-1]
avgstd = np.average(np.array(std),0)


print ("RFE Importance in Augmented Data")
#print (importances[1][0])
for idx in importances[0][0]:
    print ("Feature %s has importance in RFE Gaussian"%(fnames[idx]))

print ("Random Forest Importance")

for ti in indices:
    print ("Feature %s has probability %f" %(fnames[ti], avgimport[ti]))

print ("Augmented Data F1 Scores")

r = len(fonescore) #number of classifiers
c = len(fonescore[0])#number of folds 
X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
Y = np.ones((c,r))
Y = Y*np.array(fonescore).transpose()
augf1win = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='F-1 score',title='Testing F1 Score of Classifiers',legend=names))

for i,scores in enumerate(fonescore):
    #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],opts=dict(xlabel='Fold',ylabel='F1',title='{} Augumented F1 score'.format(names[i])))
    print ("Classifier name and F1 Score: %s : "%(names[i]),end = " "),
    print (scores) 

print ("Augmented Data Accuracy Scores")

r = len(accscore) #number of classifiers
c = len(accscore[0])#number of folds 
X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
Y = np.ones((c,r))
Y = Y*np.array(accscore).transpose()
augaccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing Accuracy of Classifiers',legend=names))

for i,scores in enumerate(accscore):
    #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],opts=dict(xlabel='Fold',ylabel='F1',title='{} Augumented Accuracy score'.format(names[i])))
    print ("Classifier name and Accuracy Score: %s : "%(names[i]), end = " "),
    print (scores) 

#viz.bar(avgimport)

accscore = np.empty((7,0)).tolist()
fonescore = np.empty((7,0)).tolist()

### Only DDC maps -- features 20:end (see create feature matrix) ******************************************** #

for train, test in rkf.split(xtrain):

    txtrain = xtrain[train]
    txtrain = txtrain[:,20:30]
    tytrain = ytrain[train]
    txtest = xtrain[test]
    txtest = txtest[:,20:30]
    tytest = ytrain[test]
   # iterate over classifiers
    for i, (name, clf) in enumerate(zip(names, classifiers)):
        

        if name == "Linear_SVM":
            estimator = RFE(clf, 5, step=1)
            estimator = estimator.fit(txtrain, tytrain.ravel())
            ypred = estimator.predict(txtest)
        if name == "RBF_SVM":
            #rbfestimator = RFE(clf, 5, step=1)
            rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
            #importances[1].append(rbfestimator.get_support(indices=True))
            ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
        else:
            #creating training matrix 
            probsz = clf.fit(txtrain, tytrain.ravel())
            ypred = clf.predict(txtest)
            thescore = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
     
        
        #viz.line(X=np.ones((1))*i,Y=[accscore],win=name,name='score',update='append')
        accscore[i].append(theaccscore)
        fonescore[i].append(thescore)


r = len(accscore) #number of classifiers
c = len(accscore[0])#number of folds 
X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
Y = np.ones((c,r))
Y = Y*np.array(accscore).transpose()
daccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing DMAP Accuracy of Classifiers',legend=names))

for i, (scores,tname) in enumerate(zip(accscore,names)):
   # viz.line(X=np.arange(1,len(scores)+1),Y=scores,opts=dict(xlabel='Fold',ylabel='Accurcy',title='{} DMAP Accuracy score'.format(tname)))
    print ("Classifier name and DMAP Accuracy Score: %s : "%(names[i]), end = " "),
    print (scores) 


print ("dmaps support", end = " ")
print ([fnames[i] for i in estimator.get_support(indices=True)])

# B MAPS ONLY

print ("BMAPS")


accscore = np.empty((7,0)).tolist()
fonescore = np.empty((7,0)).tolist()

for train, test in rkf.split(xtrain):

    txtrain = xtrain[train]
    txtrain = txtrain[:,10:20]
    tytrain = ytrain[train]
    txtest = xtrain[test]
    txtest = txtest[:,10:20]
    tytest = ytrain[test]

   # iterate over classifiers
    for i, (name, clf) in enumerate(zip(names, classifiers)):
        

        if name == "Linear_SVM":
            estimator = RFE(clf, 5, step=1)
            estimator = estimator.fit(txtrain, tytrain.ravel())
            ypred = estimator.predict(txtest)
        if name == "RBF_SVM":
            #rbfestimator = RFE(clf, 5, step=1)
            rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
            #importances[1].append(rbfestimator.get_support(indices=True))
            ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
        else:
            #creating training matrix 
            probsz = clf.fit(txtrain, tytrain.ravel())
            ypred = clf.predict(txtest)
            thescore = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
     
        
        #viz.line(X=np.ones((1))*i,Y=[accscore],win=name,name='score',update='append')
        accscore[i].append(theaccscore)
        fonescore[i].append(thescore)


print ("bmaps support", end = " ")
print ([fnames[i] for i in estimator.get_support(indices=True)])

r = len(accscore) #number of classifiers
c = len(accscore[0])#number of folds 
X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
Y = np.ones((c,r))
Y = Y*np.array(accscore).transpose()
daccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing BMAP Accuracy of Classifiers',legend=names))

for i, (scores,tname) in enumerate(zip(accscore,names)):
   # viz.line(X=np.arange(1,len(scores)+1),Y=scores,opts=dict(xlabel='Fold',ylabel='Accurcy',title='{} DMAP Accuracy score'.format(tname)))
    print ("Classifier name and BMAP Accuracy Score: %s : "%(names[i]), end = " "),
    print (scores) 

# A MAPS ONLY

print ("AMAPS")
print

accscore = np.empty((7,0)).tolist()
fonescore = np.empty((7,0)).tolist()

for train, test in rkf.split(xtrain):

    txtrain = xtrain[train]
    txtrain = txtrain[:,0:10]
    tytrain = ytrain[train]
    txtest = xtrain[test]
    txtest = txtest[:,0:10]
    tytest = ytrain[test]

   # iterate over classifiers
    for i, (name, clf) in enumerate(zip(names, classifiers)):
        

        if name == "Linear_SVM":
            estimator = RFE(clf, 5, step=1)
            estimator = estimator.fit(txtrain, tytrain.ravel())
            ypred = estimator.predict(txtest)
        if name == "RBF_SVM":
            #rbfestimator = RFE(clf, 5, step=1)
            rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
            #importances[1].append(rbfestimator.get_support(indices=True))
            ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
            thef1score = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
        else:
            #creating training matrix 
            probsz = clf.fit(txtrain, tytrain.ravel())
            ypred = clf.predict(txtest)
            thescore = f1_score(tytest.ravel(),ypred)
            theaccscore = accuracy_score(tytest.ravel(),ypred)
     
        
        #viz.line(X=np.ones((1))*i,Y=[accscore],win=name,name='score',update='append')
        accscore[i].append(theaccscore)
        fonescore[i].append(thescore)


r = len(accscore) #number of classifiers
c = len(accscore[0])#number of folds 
X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
Y = np.ones((c,r))
Y = Y*np.array(accscore).transpose()
daccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing AMAP Accuracy of Classifiers',legend=names))

for i, (scores,tname) in enumerate(zip(accscore,names)):
   # viz.line(X=np.arange(1,len(scores)+1),Y=scores,opts=dict(xlabel='Fold',ylabel='Accurcy',title='{} DMAP Accuracy score'.format(tname)))
    print ("Classifier name and AMAP Accuracy Score: %s : "%(names[i]), end = " "),
    print (scores) 


print ("amaps support", end = " ")
print ([fnames[i] for i in estimator.get_support(indices=True)])


##### TEST ON IVIM FEATURES ######




#create Feature matrix for the original and augmented files
print ("Getting feature matrix of original maps")
xtrain, ytrain = createFeatMat3(afiles, bfiles, dfiles, diff_files, perf_files, f_files)

print ("Getting feature matrix of Crop 1 maps")
c1xtrain, c1ytrain = createFeatMat3(c1afiles, c1bfiles, c1dfiles, c1diff_files, c1perf_files, c1f_files)
print ("Getting feature matrix of Crop 2 maps")
c2xtrain, c2ytrain = createFeatMat3(c2afiles, c2bfiles, c2dfiles, c2diff_files, c2perf_files, c2f_files)
print ("Getting feature matrix of Crop 3 maps")
c3xtrain, c3ytrain = createFeatMat3(c3afiles, c3bfiles, c3dfiles, c3diff_files, c3perf_files, c3f_files)
print ("Getting feature matrix of Augmented orignal maps")
augxtrain, augytrain = createFeatMat3(augafiles, augbfiles, augdfiles, augdiff_files, augperf_files, augf_files)

#full augmented feature matrix
full_aug_xtrain = np.vstack([c1xtrain,c2xtrain,c3xtrain,augxtrain])
full_aug_ytrain = np.vstack([c1ytrain.reshape(-1,1),c2ytrain.reshape(-1,1),c3ytrain.reshape(-1,1),augytrain.reshape(-1,1)])

#Random Shuffle of Augmentations

#Importance of features obtained from RFE Linear SVM, RBF_SVM, and Random Forest -- 
importances = [None] * 3
for i in range(0,3):
    importances[i] = list()
#standard deviation of importance obtained from random forest
std = list()

#scores of accuracy and f1 for each classifier
accscore = np.empty((len(classifiers),0)).tolist()
fonescore = np.empty((len(classifiers),0)).tolist()

randindx = np.random.choice(np.arange(len(full_aug_xtrain)),size=int(len(full_aug_xtrain)*0.5),replace=True)

xtrain = np.vstack((xtrain,full_aug_xtrain[randindx]))
ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain[randindx]))

#******************************

# for train, test in rkf.split(xtrain):

#     txtrain = xtrain[train]
#     tytrain = ytrain[train]
#     txtest = xtrain[test]
#     tytest = ytrain[test]

#    # iterate over classifiers
#     for i, (name, clf) in enumerate(zip(names, classifiers)):
        
   
#         #creating training matrix 
#         if name == "Linear_SVM":
#             estimator = RFE(clf, 10, step=1)
#             estimator = estimator.fit(txtrain, tytrain.ravel())
#             importances[0].append(estimator.get_support(indices=True))
#             ypred = estimator.predict(txtest)
#             thef1score = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)
#         elif name == "RBF_SVM":
#             #rbfestimator = RFE(clf, 5, step=1)
#             rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
#             #importances[1].append(rbfestimator.get_support(indices=True))
#             ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
#             thef1score = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)

#         elif name == 'Random_Forest':
#             rfestimator = RFE(clf,10,step=1)
#             rfestimator = rfestimator.fit(txtrain, tytrain.ravel())
#             #probsz = clf.fit(txtrain, tytrain.ravel())
#             importances[2].append(rfestimator.get_support(indices=True))
#             #std.append(np.std([tree.feature_importances_ for tree in clf.estimators_],
#             # axis=0))
#             ypred = rfestimator.predict(txtest)
#             thef1score = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)

#         else:
#             probsz = clf.fit(txtrain, tytrain.ravel())
#             ypred = clf.predict(txtest)
#             thef1score = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)



      
#         accscore[i].append(theaccscore)
#         fonescore[i].append(thef1score)



# #find average of feature importances from random forest
# #avgimport = np.average(np.array(importances[2]),0)
# #indices = np.argsort(avgimport)[::-1]
# #avgstd = np.average(np.array(std),0)


# print ("RFE Importance in Augmented Data")
# #print (importances[1][0])
# for idx in importances[0][0]:
#     print ("Feature %s has importance in RFE Gaussian"%(f2names[idx]))

# print ("Random Forest Importance")

# #for ti in indices:
# #    print ("Feature %s has probability %f" %(fnames[ti], avgimport[ti]))

# for idx in importances[0][3]:
#     print ("Feature %s has importance in RFE Random Forest"%(f2names[idx]))

# print ("Augmented Data F1 Scores")

# r = len(fonescore) #number of classifiers
# c = len(fonescore[0])#number of folds 
# X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
# Y = np.ones((c,r))
# Y = Y*np.array(fonescore).transpose()
# augf1win = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='F-1 score',title='Testing IVIM F1 Score of Classifiers',legend=names))

# for i,scores in enumerate(fonescore):
#     #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],opts=dict(xlabel='Fold',ylabel='F1',title='{} Augumented F1 score'.format(names[i])))
#     print ("Classifier name and F1 Score: %s : "%(names[i]),end = " "),
#     print (scores) 

# print ("Augmented Data Accuracy Scores")

# r = len(accscore) #number of classifiers
# c = len(accscore[0])#number of folds 
# X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
# Y = np.ones((c,r))
# Y = Y*np.array(accscore).transpose()
# augaccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing IVIM Accuracy of Classifiers',legend=names))

# for i,scores in enumerate(accscore):
#     #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],opts=dict(xlabel='Fold',ylabel='F1',title='{} Augumented Accuracy score'.format(names[i])))
#     print ("Classifier name and Accuracy Score: %s : "%(names[i]), end = " "),
#     print (scores) 



# #Test Individual IVIM MAPS



# print ("DIFF MMAPS")
# print

# accscore = np.empty((7,0)).tolist()
# fonescore = np.empty((7,0)).tolist()

# for train, test in rkf.split(xtrain):

#     txtrain = xtrain[train]
#     txtrain = txtrain[:,30:40]
#     tytrain = ytrain[train]
#     txtest = xtrain[test]
#     txtest = txtest[:,30:40]
#     tytest = ytrain[test]

#    # iterate over classifiers
#     for i, (name, clf) in enumerate(zip(names, classifiers)):
        

#         if name == "Linear_SVM":
#             estimator = RFE(clf, 5, step=1)
#             estimator = estimator.fit(txtrain, tytrain.ravel())
#             ypred = estimator.predict(txtest)
#         if name == "RBF_SVM":
#             #rbfestimator = RFE(clf, 5, step=1)
#             rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
#             #importances[1].append(rbfestimator.get_support(indices=True))
#             ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
#             thef1score = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)
#         else:
#             #creating training matrix 
#             probsz = clf.fit(txtrain, tytrain.ravel())
#             ypred = clf.predict(txtest)
#             thescore = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)
     
        
#         #viz.line(X=np.ones((1))*i,Y=[accscore],win=name,name='score',update='append')
#         accscore[i].append(theaccscore)
#         fonescore[i].append(thescore)


# r = len(accscore) #number of classifiers
# c = len(accscore[0])#number of folds 
# X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
# Y = np.ones((c,r))
# Y = Y*np.array(accscore).transpose()
# daccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing IVIM DIFF Accuracy of Classifiers',legend=names))

# for i, (scores,tname) in enumerate(zip(accscore,names)):
#    # viz.line(X=np.arange(1,len(scores)+1),Y=scores,opts=dict(xlabel='Fold',ylabel='Accurcy',title='{} DMAP Accuracy score'.format(tname)))
#     print ("Classifier name and IVIM DIFF Accuracy Score: %s : "%(names[i]), end = " "),
#     print (scores) 


# print ("IVIM DIFF support: ", end = " ")
# print ([fnames[i] for i in estimator.get_support(indices=True)])



# ###### IVIM PERF:


# print ("PERF MAPS ")
# print

# accscore = np.empty((7,0)).tolist()
# fonescore = np.empty((7,0)).tolist()

# for train, test in rkf.split(xtrain):

#     txtrain = xtrain[train]
#     txtrain = txtrain[:,40:50]
#     tytrain = ytrain[train]
#     txtest = xtrain[test]
#     txtest = txtest[:,40:50]
#     tytest = ytrain[test]

#    # iterate over classifiers
#     for i, (name, clf) in enumerate(zip(names, classifiers)):
        

#         if name == "Linear_SVM":
#             estimator = RFE(clf, 5, step=1)
#             estimator = estimator.fit(txtrain, tytrain.ravel())
#             ypred = estimator.predict(txtest)
#         if name == "RBF_SVM":
#             #rbfestimator = RFE(clf, 5, step=1)
#             rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
#             #importances[1].append(rbfestimator.get_support(indices=True))
#             ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
#             thef1score = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)
#         else:
#             #creating training matrix 
#             probsz = clf.fit(txtrain, tytrain.ravel())
#             ypred = clf.predict(txtest)
#             thescore = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)
     
        
#         #viz.line(X=np.ones((1))*i,Y=[accscore],win=name,name='score',update='append')
#         accscore[i].append(theaccscore)
#         fonescore[i].append(thescore)


# r = len(accscore) #number of classifiers
# c = len(accscore[0])#number of folds 
# X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
# Y = np.ones((c,r))
# Y = Y*np.array(accscore).transpose()
# daccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing IVIM PERF Accuracy of Classifiers',legend=names))

# for i, (scores,tname) in enumerate(zip(accscore,names)):
#    # viz.line(X=np.arange(1,len(scores)+1),Y=scores,opts=dict(xlabel='Fold',ylabel='Accurcy',title='{} DMAP Accuracy score'.format(tname)))
#     print ("Classifier name and IVIM PERF Accuracy Score: %s : "%(names[i]), end = " "),
#     print (scores) 


# print ("IVIM PERF support: ", end = " ")
# print ([fnames[i] for i in estimator.get_support(indices=True)])


# ### IVIM FMAPS

# print ("F MAPS ")
# print

# accscore = np.empty((7,0)).tolist()
# fonescore = np.empty((7,0)).tolist()

# for train, test in rkf.split(xtrain):

#     txtrain = xtrain[train]
#     txtrain = txtrain[:,50:]
#     tytrain = ytrain[train]
#     txtest = xtrain[test]
#     txtest = txtest[:,50:]
#     tytest = ytrain[test]

#    # iterate over classifiers
#     for i, (name, clf) in enumerate(zip(names, classifiers)):
        

#         if name == "Linear_SVM":
#             estimator = RFE(clf, 5, step=1)
#             estimator = estimator.fit(txtrain, tytrain.ravel())
#             ypred = estimator.predict(txtest)
#         if name == "RBF_SVM":
#             #rbfestimator = RFE(clf, 5, step=1)
#             rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
#             #importances[1].append(rbfestimator.get_support(indices=True))
#             ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
#             thef1score = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)
#         else:
#             #creating training matrix 
#             probsz = clf.fit(txtrain, tytrain.ravel())
#             ypred = clf.predict(txtest)
#             thescore = f1_score(tytest.ravel(),ypred)
#             theaccscore = accuracy_score(tytest.ravel(),ypred)
     
        
#         #viz.line(X=np.ones((1))*i,Y=[accscore],win=name,name='score',update='append')
#         accscore[i].append(theaccscore)
#         fonescore[i].append(thescore)


# r = len(accscore) #number of classifiers
# c = len(accscore[0])#number of folds 
# X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
# Y = np.ones((c,r))
# Y = Y*np.array(accscore).transpose()
# daccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing IVIM F Accuracy of Classifiers',legend=names))

# for i, (scores,tname) in enumerate(zip(accscore,names)):
#    # viz.line(X=np.arange(1,len(scores)+1),Y=scores,opts=dict(xlabel='Fold',ylabel='Accurcy',title='{} DMAP Accuracy score'.format(tname)))
#     print ("Classifier name and IVIM F Accuracy Score: %s : "%(names[i]), end = " "),
#     print (scores) 


# print ("IVIM F support: ", end = " ")
# print ([fnames[i] for i in estimator.get_support(indices=True)])



# ####### More Feature Selection

# from sklearn.metrics import roc_curve, auc
# from scipy import interp
# from sklearn.svm import LinearSVC

# tprs = np.empty((7,0)).tolist()
# fprs = np.empty((7,0)).tolist()
# aucs = np.empty((7,0)).tolist()

# accscore = np.empty((7,0)).tolist()
# fonescore = np.empty((7,0)).tolist()

# for train, test in rkf.split(xtrain):

#     txtrain = xtrain[train]
#     tytrain = ytrain[train]
#     txtest = xtrain[test]
#     tytest = ytrain[test]

#    # iterate over classifiers
#     for i, (name, clf) in enumerate(zip(names, classifiers)):
        
#         clf2 = Pipeline([('feature_selection', SelectFromModel(LinearSVC(C=1,penalty="l1",dual=False,max_iter=5000))),
#         ('classification', clf)])
#         probsz = clf2.fit(txtrain, tytrain.ravel()).predict_proba(txtest)
#         ypred = clf2.predict(txtest)
#         thef1score = f1_score(tytest.ravel(),ypred)
#         theaccscore = accuracy_score(tytest.ravel(),ypred)
#         fpr, tpr, thresholds = roc_curve(tytest, probsz[:,1])
#         #tprs[i].append(interp(mean_fpr,fpr,tpr))
#         tprs[i].append(tpr)
#         fprs[i].append(fpr)

#         roc_auc = auc(fpr,tpr)
#         aucs[i].append(roc_auc)
#         accscore[i].append(theaccscore)
#         fonescore[i].append(thef1score)



# print ("Augmented Data F1 Scores")

# r = len(fonescore) #number of classifiers
# c = len(fonescore[0])#number of folds 
# X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
# Y = np.ones((c,r))
# Y = Y*np.array(fonescore).transpose()
# augf1win = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='F-1 score',title='Testing IVIM F1 Score of Classifiers',legend=names))

# for i,scores in enumerate(fonescore):
#     #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],opts=dict(xlabel='Fold',ylabel='F1',title='{} Augumented F1 score'.format(names[i])))
#     print ("Classifier name and F1 Score: %s : "%(names[i]),end = " "),
#     print (scores) 

# print ("Augmented Data Accuracy Scores")

# r = len(accscore) #number of classifiers
# c = len(accscore[0])#number of folds 
# X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
# Y = np.ones((c,r))
# Y = Y*np.array(accscore).transpose()
# augaccwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title='Testing IVIM Accuracy of Classifiers',legend=names))

# for i,scores in enumerate(accscore):
#     #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],opts=dict(xlabel='Fold',ylabel='F1',title='{} Augumented Accuracy score'.format(names[i])))
#     print ("Classifier name and Accuracy Score: %s : "%(names[i]), end = " "),
#     print (scores) 


# print ("ROC CURVES for IVIM")

# c = len(max(fprs,key=len)) #number of folds
# for i in range(0,len(names)):
#     r = len(max(fprs[i],key=len)) #finds the maximum length of an array within a set of arrays
#     X = np.ones((r,c)) #create initial matrix fprs x folds
#     Y = np.ones((r,c))
#     fold_names = []
#     #Problem every fold has a different number of FPRS ---- ???
#     for j in range(0,c):
#         X[0:len(fprs[i][j]), j] = fprs[i][j]
#         Y[0:len(fprs[i][j]), j] = tprs[i][j]
#         temp = 'Fold {0:5d} - AUC {1:.2f}'.format(j, aucs[i][j])
#         fold_names.append(temp)
#     viz.line(X=X,Y=Y,opts=dict(xlabel='fpr',ylabel='tpr',title='Testing IVIM ROC of Classifiers {}'.format(names[i]),legend=fold_names))




# ### Pipeline for GridSearch 


# #****** GRID SEARCH FOR OPTIMAL PARAMATERS for Gradient Boost Classifier   ********



# from sklearn.ensemble import GradientBoostingClassifier
# print ("GRID SEARCH")




# """ tuned_parameters = {"loss": ["deviance"], "learning_rate": [0.01, .025, .05, .075, 0.1, 0.15, 0.2], "min_samples_split": np.linspace(0.1,0.5,8),"min_samples_leaf": np.linspace(0.1,0.5,8),
# "max_depth": [3,5,8],"max_features":["log2", "sqrt"],"criterion": ["friedman_mse"],"subsample": [0.5, 0.618, 0.8, 0.85, 1.0], "n_estimators":[100,250,500]}


# cv_grid = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=10, scoring='roc_auc', n_jobs=-1, verbose=True)
# cv_grid.fit(xtrain, ytrain.ravel())


# print ("CV_Grid score: ", end = " ") 
# print (cv_grid.score(xtrain, ytrain.ravel()))

# print ("CV_Grid pest parameters: ", end = " ") 
# print (cv_grid.best_params_)

# b_estimator = cv_grid.best_estimator_
# b_estimator.fit(xtrain,ytrain.ravel())

# b_feats = b_estimator.feature_importances_

# #viz.bar(X=b_feats,opts=dict(stacked=False,rownames=f2names))



# np.save("bestfeats", [b_feats, f2names])
#  """

#*******************


after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': 'sqrt', 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
'criterion': 'friedman_mse', 'n_estimators': 100}

model = GradientBoostingClassifier(**after_tuned)

#xtrain, xtest, ytrain, ytest = train_test_split(xtrain,ytrain.ravel(),test_size=0.3,random_state=9)

#score = model.fit(xtrain,ytrain).predict_proba(xtest)


#fpr, tpr, thresholds = roc_curve(ytest, score[:,1])
#ypred = model.predict(xtest)
#roc_auc = auc(fpr,tpr)

#thef1score = f1_score(ytest,ypred)
#theaccscore = accuracy_score(ytest,ypred)

#fold_names = ['AUC: {0:.2f} F1: {1:.2f} Acc: {2:.2f}'.format(roc_auc, thef1score, theaccscore)]

tprs = []
accscore = []
fonescore = []
aucs = []
mean_fpr = np.linspace(0,1,100)
#all_fpr = []
#all_tpr = []
for train, test in rkf.split(xtrain):

    txtrain = xtrain[train]
    tytrain = ytrain[train]
    txtest = xtrain[test]
    tytest = ytrain[test]

    probz = model.fit(txtrain,tytrain.ravel()).predict_proba(txtest)
    fpr, tpr, thresholds = roc_curve(tytest.ravel(),probz[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    aucs.append(auc(fpr,tpr))
    ypred = model.predict(txtest)
    fonescore.append(f1_score(tytest.ravel(),ypred))
    accscore.append(accuracy_score(tytest.ravel(),ypred))
    

mean_tpr = np.mean(tprs,axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

tprs.append(mean_tpr)
aucs.append(mean_auc)
accscore.append(np.mean(accscore))
fonescore.append(np.mean(fonescore))


print ("ROC CURVES for Best Estimator")

c = len(tprs) #number of folds
r = len(max(tprs,key=len)) #finds the maximum length of an array within a set of arrays
X = np.ones((r,c)) #create initial matrix fprs x folds
Y = np.ones((r,c))
fold_names = []

for i in range(0,c):
    X[0:len(tprs[i]),i] = mean_fpr
    Y[0:len(tprs[i]),i] = tprs[i]
    temp = 'Fold {0:5d} - AUC: {1:.2f} F1: {2:.2f} Acc: {3:.2f}'.format(i, aucs[i], fonescore[i], accscore[i])
    fold_names.append(temp)


viz.line(X=X,Y=Y,opts=dict(xlabel='fpr',ylabel='tpr',title='ROC of Best Estimator',legend=fold_names))

#viz.line(X=fpr,Y=tpr,opts=dict(xlabel='fpr',ylabel='tpr',title='ROC of Best Gradient Boost Estimator {}'.format(names[i]),legend=fold_names))



print ("COMPLETED WOO")