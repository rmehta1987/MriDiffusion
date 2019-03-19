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

    
#Feature Names, a10 represents 10th percentile and so on, note that 50th percentile is equivalent to median of histogram
fnames = ["a10", "a25", "a50", "a75", "aquartile", "amean", "avar", "akurt", 
"askew",  "b10", "b25", "b50", "b75", "bquartile", "bmean", "bvr", "bkurt", 
"bskew", "d10", "d25", "d50", "d75", "dquartile","dmean",  "dvra", "dkurt", "dskew"]         

f2names = ["a10", "a25", "a50", "a75", "aquartile", "amean", "avar", "akurt", 
"askew",  "b10", "b25", "b50", "b75", "bquartile", "bmean", "bvar", "bkurt", 
"bskew", "d10", "d25", "d50", "d75", "dquartile","dmean", "dvar", "dkurt", "dskew",
"diff10", "diff25", "diff50", "diff75", "diffquartile", "diffmean", "diffvar", "diffkurt", "diffskew",
"perf10", "perf25", "perf50", "perf75", "perfquartile", "perfmean", "perfvar", "perfkurt", "perfskew",
"f10", "f25", "f50", "f75", "fquartile", "fmean", "fvar", "fkurt", "fskew"]       

print (len(f2names))

#This function gets gets all the maps and loads them into a list, can be modified if more maps are added later
def getFiles(file_path, name):
    '''@file_path: where files are stored
        @name: name of the individual maps, maps should be .npy format and file names are mapname*.npy'''

    afiles = sorted(glob.glob('%s/%s*.npy'%(file_path[0],name[0])))
    bfiles = sorted(glob.glob('%s/%s*.npy'%(file_path[0],name[1])))
    dfiles = sorted(glob.glob('%s/%s*.npy'%(file_path[0],name[2])))
    difffiles = sorted(glob.glob('%s/%s*.npy'%(file_path[1],name[3])))
    perffiles = sorted(glob.glob('%s/%s*.npy'%(file_path[1],name[4])))
    ffiles = sorted(glob.glob('%s/%s*.npy'%(file_path[1],name[5])))
    lafiles = []
    lbfiles = []
    ldfiles = []
    ldifffiles = []
    lperffiles = []
    lffiles = []
    print ("obtaining files in the getfiles function")
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles,bfiles,dfiles,difffiles,perffiles,ffiles)):
        lafiles.append(np.load(a))
        lbfiles.append(np.load(b))
        ldfiles.append(np.load(d))
        ldifffiles.append(np.load(e))
        lperffiles.append(np.load(f))
        lffiles.append(np.load(g))
    
    return lafiles, lbfiles, ldfiles, ldifffiles, lperffiles, lffiles

#This function is meant for the original maps, alpha, beta, ddc, see createFeatMat3, where it is created for given maps
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

    print ("finished creating Feature and Label Matrix in createFeatMat2")
    return xtrain, ytrain

#IVIM FEATURE MATRIX, creates a feature matrix of all the maps, more can be added if necessary
def createFeatMat3(afiles, bfiles, dfiles, diff_files, perf_files, f_files):
    '''creates a feature matrix from the different set of files
    @param afiles = features of amaps  - arranged as the map, label, patient name, histogram features, histogram in a numpy array
    @param bfiles = features of bmaps
    @param dfiles = features of dmaps'''

    xtrain = np.zeros((len(afiles),len(afiles[0][3])*6)) #Feature Matrix [a-features, b-features, ddc-features, diff-features, perf-features, f-features]
    ytrain = np.zeros((len(afiles),),dtype=np.int) #label matrix
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles,bfiles,dfiles,diff_files, perf_files, f_files)):
        xtrain[i] = np.hstack((a[3], b[3], d[3], e[3], f[3], g[3]))
        ytrain[i] = a[1]

    print ("finished creating Feature and Label Matrix createFeatMat3, such that original and IVIM features are together")
    return xtrain, ytrain


#Starts visdom, the visualizer for graphs
viz = Visdom()

#Initialize Classifier hyperparameters
classifiers = [
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()]

#Repeated Cross Validation
rkf = RepeatedKFold(n_splits=3, n_repeats=2)

#Get all the Files
print ("Getting Files containing features of original maps")
file_path = ['maxminFeat', 'maxminFeatIvim']
name = ['mm_apad_feat','mm_bpad_feat','mm_dpad_feat', 'mm_diffpad_feat', 'mm_perfpad_feat', 'mm_fpad_feat']
afiles, bfiles, dfiles, diff_files, perf_files, f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Crop 1")
name = ['cropaug1_alpha_feat','cropaug1_beta_feat','cropaug1_ddc_feat', 'cropaug1_diff_feat', 'cropaug1_perf_feat', 'cropaug1_f_feat']
c1afiles, c1bfiles, c1dfiles, c1diff_files, c1perf_files, c1f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Crop 2")
name = ['cropaug2_alpha_feat','cropaug2_beta_feat','cropaug2_ddc_feat', 'cropaug2_diff_feat', 'cropaug2_perf_feat', 'cropaug2_f_feat']
c2afiles, c2bfiles, c2dfiles, c2diff_files, c2perf_files, c2f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Crop 3")
name = ['cropaug3_alpha_feat','cropaug3_beta_feat','cropaug3_ddc_feat', 'cropaug3_diff_feat', 'cropaug3_perf_feat', 'cropaug3_f_feat']
c3afiles, c3bfiles, c3dfiles, c3diff_files, c3perf_files, c3f_files = getFiles(file_path,name)

print ("Getting files of augmented maps -- Original Augmented Rotated")
name = ['ogaug_alpha_feat','ogaug_beta_feat','ogaug_ddc_feat', 'ogaug_diff_feat', 'ogaug_perf_feat', 'ogaug_f_feat']
augafiles, augbfiles, augdfiles, augdiff_files, augperf_files, augf_files = getFiles(file_path,name)

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

def runClassifiers(xtrain, ytrain):
    '''@xtrain feature matrix
        @ytrain label matrix'''

    #Importance of features obtained from RFE Linear SVM, RBF_SVM, and Random Forest -- 
    importances = [None] * 3
    for i in range(0,3):
        importances[i] = list()

    #standard deviation of importance obtained from random forest
    std = list()

    #scores of accuracy and f1 for each classifier
    accscore = np.empty((len(classifiers),0)).tolist()
    fonescore = np.empty((len(classifiers),0)).tolist()

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
                ypred = estimator.predict(txtest)
                importances[0].append(estimator.get_support(indices=True))
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
            elif name == "RBF_SVM":
                rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
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
        
    return accscore, fonescore, std, importances

def visualizeResults(accscore, fonescore, std, importances, allnames, thetitle):
    '''@accscore is the accuracy score of the individual classifiers
        @fonescore is the F1 score of the individual classifiers
        @std standrad deviation of feature importances **CURRENTLY NOT IMPLEMENTED**
        @importances importances of classifiers that can return feature importances, only index 0 and 2 are currently implemetned
        where importances[0] is SVM RFE Linear and importances[2] is Random Forest Importance'''

    #find average of feature importances from random forest
    avgimport = np.average(np.array(importances[2]),0)
    indices = np.argsort(avgimport)[::-1]
    avgstd = np.average(np.array(std),0)


    print ("RFE Importance ", end = " ")
    print ([allnames[i] for i in importances[0][-1]])


    print ("Random Forest Importance", end = " ")
    print ([(allnames[i],avgimport[i]) for i in indices])

    print ("F1 Scores")

    r = len(fonescore) #number of classifiers
    c = len(fonescore[0])#number of folds 
    X = np.tile(np.arange(1,c+1).reshape(-1,1),(1,r))
    Y = np.ones((c,r))
    Y = Y*np.array(fonescore).transpose()
    newtitle = 'F1 Score ' + thetitle
    f1win = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='F-1 score',title=newtitle,legend=names))

    for i,scores in enumerate(fonescore):
        #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=orgf1score,opts=dict(xlabel='Fold',ylabel='F1',title='{} Original F1 score'.format(names[i])))
        #viz.line(X=np.arange(1,len(scores)+1),Y=scores,name=names[i],win=orgf1score,opts=dict(xlabel='Fold',ylabel='F1'),update='append')
        print ("Classifier name and F1 Score: %s : "%(names[i]), end = " ")
        print (scores) 

    print ("Accuracy Scores")

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
    newtitle = 'Accuracy ' + thetitle
    accwin = viz.line(X=X,Y=Y,opts=dict(xlabel='Fold',ylabel='Accuracy',title=newtitle,legend=names))   



print
print ("Testing Original Data")

atitle = "Original Data"
accscore, fonescore, std, importances = runClassifiers(xtrain, ytrain)
visualizeResults(accscore, fonescore, std, importances, fnames, atitle)


print
print ("Testing Improvement of Augmented Data")

#Random Shuffle of Augmentations

randindx = np.random.choice(np.arange(len(full_aug_xtrain)),size=int(len(full_aug_xtrain)*0.5),replace=True)
xtrain = np.vstack((xtrain,full_aug_xtrain[randindx]))
ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain[randindx]))

#Run Classifier for Augumented Data
atitle = "Augumented Data"

accscore, fonescore, std, importances = runClassifiers(xtrain, ytrain)
visualizeResults(accscore, fonescore, std, importances, fnames, atitle)


### Only DDC maps -- features 20:30 (see create feature matrix) ******************************************** #
atitle = "DMAPS"

accscore, fonescore, std, importances = runClassifiers(xtrain[:,18:27], ytrain)
visualizeResults(accscore, fonescore, std, importances, fnames[18:27], atitle)


### Only B maps -- features 10:20 (see create feature matrix) ******************************************** #
print ("BMAPS")
atitle = "BMAPS"
#Run Classifiers
accscore, fonescore, std, importances = runClassifiers(xtrain[:,9:18], ytrain)
visualizeResults(accscore, fonescore, std, importances, fnames[9:18], atitle)



# A MAPS ONLY
atitle = "AMAPS"
print ("AMAPS")
accscore, fonescore, std, importances = runClassifiers(xtrain[:,0:9], ytrain)
visualizeResults(accscore, fonescore, std, importances, fnames[0:9], atitle)


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

randindx = np.random.choice(np.arange(len(full_aug_xtrain)),size=int(len(full_aug_xtrain)*0.5),replace=True)

xtrain = np.vstack((xtrain,full_aug_xtrain[randindx]))
ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain[randindx]))

atitle = "Org and IVIM Maps"
accscore, fonescore, std, importances = runClassifiers(xtrain, ytrain)
visualizeResults(accscore, fonescore, std, importances, f2names, atitle)

##################################################Test Individual IVIM MAPS##################################################


print ("DIFF MMAPS")
print
atitle = "DIFF"
accscore, fonescore, std, importances = runClassifiers(xtrain[:,27:36], ytrain)
visualizeResults(accscore, fonescore, std, importances, f2names[27:36], atitle)


# ###### IVIM PERF:
print ("PERF MMAPS")
print
atitle = "PERF"
accscore, fonescore, std, importances = runClassifiers(xtrain[:,36:45], ytrain)
visualizeResults(accscore, fonescore, std, importances, f2names[36:45], atitle)

# ### IVIM FMAPS

print ("F MAPS ")
print
atitle = "F_MAPS"
accscore, fonescore, std, importances = runClassifiers(xtrain[:,45:], ytrain)
visualizeResults(accscore, fonescore, std, importances, f2names[45:], atitle)

def runGridSearch(tuned_parameters,grid_classifier, xtrain, ytrain, allnames):

    cv_grid = GridSearchCV(grid_classifier, tuned_parameters, cv=10, scoring='roc_auc', n_jobs=-1, verbose=True)
    cv_grid.fit(xtrain, ytrain.ravel())

    print ("CV_Grid score: ", end = " ") 
    print (cv_grid.score(xtrain, ytrain.ravel()))

    print ("CV_Grid pest parameters: ", end = " ") 
    print (cv_grid.best_params_)

    b_estimator = cv_grid.best_estimator_
    b_estimator.fit(xtrain,ytrain.ravel())

    b_feats = b_estimator.feature_importances_
    np.save("bestfeats", [b_feats, allnames])
    np.save("bestparams", cv_grid.best_params_)
    viz.bar(X=b_feats,opts=dict(stacked=False,rownames=allnames))

    return b_estimator, cv_grid.best_params_

def oneModel(themodel, xtrain, ytrain):


    tprs = []
    accscore = []
    fonescore = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)

    for train, test in rkf.split(xtrain):

        txtrain = xtrain[train]
        tytrain = ytrain[train]
        txtest = xtrain[test]
        tytest = ytrain[test]

        probz = themodel.fit(txtrain,tytrain.ravel()).predict_proba(txtest)
        fpr, tpr, thresholds = roc_curve(tytest.ravel(),probz[:,1])
        tprs.append(interp(mean_fpr,fpr,tpr))
        tprs[-1][0]=0.0
        aucs.append(auc(fpr,tpr))
        ypred = themodel.predict(txtest)
        fonescore.append(f1_score(tytest.ravel(),ypred))
        accscore.append(accuracy_score(tytest.ravel(),ypred))

    return tprs, aucs, mean_fpr, accscore, fonescore, themodel
    
def visRocCurve(tprs, aucs, mean_fpr, accscore, fonescore, model, allnames):

    mean_tpr = np.mean(tprs,axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    tprs.append(mean_tpr)
    aucs.append(mean_auc)
    accscore.append(np.mean(accscore))
    fonescore.append(np.mean(fonescore))


    print ("ROC CURVES")

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


    viz.line(X=X,Y=Y,opts=dict(xlabel='fpr',ylabel='tpr',title='ROC',legend=fold_names))
    b_feats = model.feature_importances_

    indices = np.argsort(b_feats)
    temp = np.array(allnames)[indices]

    viz.bar(X=np.sort(b_feats),opts=dict(stacked=False,rownames=temp.tolist()))


# ### Pipeline for GridSearch 


# #****** GRID SEARCH FOR OPTIMAL PARAMATERS for Gradient Boost Classifier   ********
# print ("GRID SEARCH")

# """ tuned_parameters = {"loss": ["deviance"], "learning_rate": [0.01, .025, .05, .075, 0.1, 0.15, 0.2], "min_samples_split": np.linspace(0.1,0.5,8),"min_samples_leaf": np.linspace(0.1,0.5,8),
# "max_depth": [3,5,8],"max_features":["log2", "sqrt"],"criterion": ["friedman_mse"],"subsample": [0.5, 0.618, 0.8, 0.85, 1.0], "n_estimators":[100,250,500]}

#grid_classifier = GradientBoostingClassifier()

#the_estimator, after_tuned = runGridSearch(tuned_parameters,grid_classifier, xtrain, ytrain, f2names):

#*******************


after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': 'sqrt', 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
'criterion': 'friedman_mse', 'n_estimators': 100}

model = GradientBoostingClassifier(**after_tuned)
tprs, aucs, mean_fpr, accscore, fonescore, model = oneModel(model, xtrain, ytrain)
visRocCurve(tprs, aucs, mean_fpr, accscore, fonescore, model, f2names)

print ("COMPLETED WOO")