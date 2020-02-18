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
from sklearn.utils import resample
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
from boruta import BorutaPy
import matplotlib as mpl
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import sys


#Global Variables

#Classifier Names
names = ["Linear_SVM", "RBF_SVM", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "AdaBoost",
         "Naive_Bayes", "Gradient_Boosted"]

#Pretty classifier Names
c_names = ["Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes", "Gradient Boosted"]

    
#Feature Names, a10 represents 10th percentile and so on, note that 50th percentile is equivalent to median of histogram
fnames = ["a10", "a25", "a50", "a75", "aquartile", "amean", "avar", "akurt", 
"askew",  "b10", "b25", "b50", "b75", "bquartile", "bmean", "bvr", "bkurt", 
"bskew", "d10", "d25", "d50", "d75", "dquartile","dmean",  "dvra", "dkurt", "dskew"]         

f2names = [r'$\alpha_{10}$', r'$\alpha_{Q1}$', r'$\alpha_{median}$', r'$\alpha_{Q3}$', r'$\alpha_{IQR}$', r'$\alpha_{mean}$', r'$\alpha_{variance}$', r'$\alpha_{kurtosis}$', 
r'$\alpha_{skewnewss}$',  r'$\beta_{10}$', r'$\beta_{Q1}$', r'$\beta_{median}$', r'$\beta_{Q3}$', r'$\beta_{IQR}$', r'$\beta_{mean}$', r'$\beta_{variance}$', r'$\beta_{kurtosis}$', 
r'$\beta_{skewness}$', r'$D^{m}_{10}$', r'$D^{m}_{Q1}$', r'$D^{m}_{median}$', r'$D^{m}_{Q3}$', r'$D^{m}_{IQR}$', r'$D^{m}_{mean}$', r'$D^{m}_{variance}$', r'$D^{m}_{kurtosis}$', r'$D^{m}_{skewness}$',
r'$D^{diff}_{10}$', r'$D^{diff}_{Q1}$', r'$D^{diff}_{median}$', r'$D^{diff}_{Q3}$', r'$D^{diff}_{IQR}$', r'$D^{diff}_{mean}$',r'$D^{diff}_{variance}$', r'$D^{diff}_{kurtosis}$', r'$D^{diff}_{skewness}$',
r'$D^{perf}_{10}$', r'$D^{perf}_{Q1}$', r'$D^{perf}_{median}$', r'$D^{perf}_{Q3}$', r'$D^{perf}_{IQR}$', r'$D^{perf}_{mean}$', r'$D^{perf}_{variance}$',r'$D^{perf}_{kurtosis}$', r'$D^{perf}_{skewness}$',
r'$F_{10}$', r'$F_{Q1}$', r'$F_{median}$', r'$F_{Q3}$', r'$F_{IQR}$', r'$F_{mean}$', r'$F_{variance}$', r'$F_{kurtosis}$', r'$F_{skewness}$']       

# Starts visdom, the visualizer for graphs
# viz = Visdom()

# Initialize Classifier hyperparameters
classifiers = [
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(max_depth=5, n_estimators=10, max_features=1)]


#Repeated Cross Validation
rkf = RepeatedKFold(n_splits=5, n_repeats=3)


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

    xtrain = np.zeros((len(afiles),len(afiles[0][3])*6))  # Feature Matrix [a-features, b-features, ddc-features, diff-features, perf-features, f-features]
    ytrain = np.zeros((len(afiles),),dtype=np.int)  # label matrix
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles,bfiles,dfiles,diff_files, perf_files, f_files)):
        xtrain[i] = np.hstack((a[3], b[3], d[3], e[3], f[3], g[3]))
        ytrain[i] = a[1]

    print ("finished creating Feature and Label Matrix createFeatMat3, such that original and IVIM features are together")
    return xtrain, ytrain


def runGridSearch(tuned_parameters,grid_classifier, xtrain, ytrain, allnames, classifer_name):

    cv_grid = GridSearchCV(grid_classifier, tuned_parameters, cv=10, scoring='roc_auc', n_jobs=-1, verbose=True)
    cv_grid.fit(xtrain, ytrain.ravel())

    print ("CV_Grid score: ", end = " ") 
    print (cv_grid.score(xtrain, ytrain.ravel()))

    print ("CV_Grid pest parameters: ", end = " ") 
    print (cv_grid.best_params_)

    b_estimator = cv_grid.best_estimator_
    b_estimator.fit(xtrain,ytrain.ravel())

    b_feats = b_estimator.feature_importances_
    np.save("bestfeats_{}".format(classifer_name), [b_feats, allnames])
    np.save("bestparams_{}".format(classifer_name), cv_grid.best_params_)
    viz.bar(X=b_feats,opts=dict(stacked=False,rownames=allnames))

    return b_estimator, cv_grid.best_params_

def oneModel(themodel, xtrain, ytrain):


    tprs = []
    accscore = []
    fonescore = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    b_feats = []

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
        

    return tprs, aucs, mean_fpr, accscore, fonescore, themodel, b_feats
    
def visRocCurve(tprs, aucs, mean_fpr, accscore, fonescore, model, allnames, b_feats):

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


    viz.line(X=X,Y=Y,opts=dict(xlabel='fpr',ylabel='tpr',title='Gradient Boosted ROC',legend=fold_names))
    
    #calculate average of feature probabilities and standard deviations

    #indices = np.argsort(b_feats[2])
    #temp = np.array(allnames)[indices]

    #viz.bar(X=np.sort(b_feats[2]),opts=dict(stacked=False,rownames=temp.tolist()))

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
    tprs = np.empty((len(classifiers),0)).tolist()
    aucs = np.empty((len(classifiers),0)).tolist()
    
    mean_fpr = np.linspace(0,1,100)
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
                probROC = estimator.predict_proba(txtest)
                importances[0].append(estimator.get_support(indices=True))
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
                
            elif name == "RBF_SVM":
                rbfestimator = clf.fit(txtrain[:,estimator.get_support(indices=True)], tytrain.ravel())
                ypred = clf.predict(txtest[:,estimator.get_support(indices=True)])
                probROC = clf.predict_proba(txtest[:,estimator.get_support(indices=True)])
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
            elif name == 'Random_Forest':
                probsz = clf.fit(txtrain, tytrain.ravel())
                timportance = clf.feature_importances_
                importances[2].append(timportance)
                std.append(np.std([tree.feature_importances_ for tree in clf.estimators_],
                axis=0))
                ypred = clf.predict(txtest)
                probROC = clf.predict_proba(txtest)
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
            else:
                probsz = clf.fit(txtrain, tytrain.ravel())
                ypred = clf.predict(txtest)
                probROC = clf.predict_proba(txtest)
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
                
            
            fpr, tpr, thresholds = roc_curve(tytest.ravel(),probROC[:,1])
            tprs[i].append(interp(mean_fpr,fpr,tpr))
            tprs[i][-1][0]=0.0
            aucs[i].append(auc(fpr,tpr))
       
            accscore[i].append(theaccscore)
            fonescore[i].append(thef1score)
        
    return accscore, fonescore, std, importances, tprs, aucs

   
def make_errors3(allys, cis, xs=None, color=None, name=None, title=None, l_names=None):
    '''
    See: https://github.com/facebookresearch/visdom/issues/155 and https://github.com/facebookresearch/visdom/issues/18
    on how to directly send plots to plotly and plotting error bars in plotly: https://plot.ly/python/continuous-error-bars/
    Args:
        allys: A ROC curve for each classifier
        CIS: A lower cis[0] and upper cis[1] binomial confidence interval    
    '''

    #create a range for x-values

    xs = np.linspace(0,1,100).tolist()
    name = "ROC"
    ylow = cis[0]
    yup = cis[1]
    allfills = ['rgba(0,176,246,0.2)','rgba(0,100,80,0.2)','rgba(231,107,243,0.2)','rgba(2,207,104,0.2)','rgba(143, 19, 131,0.2)','rgba(113,176,255 ,0.2)',
                'rgba(243,135,137,0.2)','rgba(68, 68, 68,0.5)' ]
    linecolor = ['rgb(0,176,246)','rgb(0,100,80)','rgb(231,107,243)','rgb(68, 68, 68)','rgb(143, 19, 131)','rgb(113,176,255 )',
                'rgb(243,135,137)','rgb(68, 68, 68)']
    layout=dict(title=title, xaxis={'title':'FPR'}, yaxis={'title':'TPR'})
    err_traces = []
    for i, yavg in enumerate(allys):

        err_traces.append(dict(x=xs, y=yavg.tolist(), mode='lines', type='line', name=l_names[i],
                line=dict(color=linecolor[i])))
        err_traces.append(dict(
                x=xs + xs[::-1],
                y=(yup[i]).tolist() + (ylow[i]).tolist()[::-1],
                fill='tozerox',
                fillcolor=allfills[i],
                line=dict(color='transparent'), type='line',showlegend=False,
                name=l_names[i] + "_error",
            ))
        
    viz._send({'data': err_traces, 'layout': layout, 'win': 'mywin_{}'.format(title)})


def make_errors4(allys, cis, opt_tpr, xs=None, color=None, name=None, title=None, l_names=None):
    '''
    See: https://github.com/facebookresearch/visdom/issues/155 and https://github.com/facebookresearch/visdom/issues/18
    on how to directly send plots to plotly and plotting error bars in plotly: https://plot.ly/python/continuous-error-bars/
    Args:
        allys: A ROC curve for each classifier
        CIS: A lower cis[0] and upper cis[1] binomial confidence interval    
    '''

    #create a range for x-values

    xs = np.linspace(0,1,100).tolist()
    name = "ROC"
    ylow = cis[0]
    yup = cis[1]
    allfills = ['rgba(0,176,246,0.2)','rgba(0,100,80,0.2)','rgba(231,107,243,0.2)','rgba(2,207,104,0.2)','rgba(143, 19, 131,0.2)','rgba(113,176,255 ,0.2)',
                'rgba(243,135,137,0.2)','rgba(68, 68, 68,0.5)' ]
    linecolor = ['rgb(0,176,246)','rgb(0,100,80)','rgb(231,107,243)','rgb(68, 68, 68)','rgb(143, 19, 131)','rgb(113,176,255 )',
                'rgb(243,135,137)','rgb(68, 68, 68)']
    layout=dict(title=title, xaxis={'title':'TPR'}, yaxis={'title':'FPR'})
    err_traces = []
    for i, yavg in enumerate(allys):

        err_traces.append(dict(x=xs, y=yavg.tolist(), mode='lines', type='line', name=l_names[i],
                line=dict(color=linecolor[i])))
        err_traces.append(dict(
                x=xs + xs[::-1],
                y=(yup[i]).tolist() + (ylow[i]).tolist()[::-1],
                fill='tozerox',
                fillcolor=allfills[i],
                line=dict(color='transparent'), type='line',showlegend=False,
                name=l_names[i] + "_error",
            ))
    
    #viz.line(X=fpr,Y=tpr,opts=dict(xlabel='fpr',ylabel='tpr',title='Gradient Boosted ROC'))
    
    err_traces.append(dict(x=xs, y=opt_tpr.tolist(), mode='lines', type='line', name="Optimal Gradient Boosted",
                line=dict(color="rgb(0,0,0)")))

    viz._send({'data': err_traces, 'layout': layout, 'win': 'mywin'})

    
   


def runClassifiers2(xtrain, ytrain):
    ''' This will also now do random undersampling of majority class
    
        @xtrain feature matrix
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
    tprs = np.empty((len(classifiers),0)).tolist()
    aucs = np.empty((len(classifiers),0)).tolist()
    
    mean_fpr = np.linspace(0,1,100)
    for train, test in rkf.split(xtrain):

        txtrain = xtrain[train]
        tytrain = ytrain[train]
        txtest = xtrain[test]
        tytest = ytrain[test]

    # iterate over classifiers
        for i, (name, clf) in enumerate(zip(names, classifiers)):
            
    
            #creating training matrix 
            if name == "Linear_SVM":
                estimator = make_pipeline(RandomUnderSampler(sampling_strategy='auto',replacement=True),RFE(clf,10,step=1))
                #estimator = RFE(clf, 10, step=1)
                estimator.fit(txtrain, tytrain.ravel())
                ypred = estimator.predict(txtest)
                probROC = estimator.predict_proba(txtest)
                importances[0].append(estimator.steps[1][1].get_support(indices=True))
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
                
            elif name == "RBF_SVM":
                rbfpipeline = make_pipeline(RandomUnderSampler(sampling_strategy='auto',replacement=True),clf)
                rbfpipeline.fit(txtrain[:,estimator.steps[1][1].get_support(indices=True)], tytrain.ravel())
                ypred = rbfpipeline.predict(txtest[:,estimator.steps[1][1].get_support(indices=True)])
                probROC = rbfpipeline.predict_proba(txtest[:,estimator.steps[1][1].get_support(indices=True)])
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
            elif name == 'Random_Forest':
                estimator = make_pipeline(RandomUnderSampler(sampling_strategy='auto',replacement=True),clf)
                estimator.fit(txtrain, tytrain.ravel())
                timportance = estimator.steps[1][1].feature_importances_
                importances[2].append(timportance)
                std.append(np.std([tree.feature_importances_ for tree in estimator.steps[1][1].estimators_],
                axis=0))
                ypred = estimator.predict(txtest)
                probROC = estimator.predict_proba(txtest)
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
            else:
                estimator = make_pipeline(RandomUnderSampler(sampling_strategy='auto',replacement=True),clf)
                probsz = estimator.fit(txtrain, tytrain.ravel())
                ypred = estimator.predict(txtest)
                probROC = estimator.predict_proba(txtest)
                thef1score = f1_score(tytest.ravel(),ypred)
                theaccscore = accuracy_score(tytest.ravel(),ypred)
                
            
            fpr, tpr, thresholds = roc_curve(tytest.ravel(),probROC[:,1])
            tprs[i].append(interp(mean_fpr,fpr,tpr))
            tprs[i][-1][0]=0.0
            aucs[i].append(auc(fpr,tpr))
       
            accscore[i].append(theaccscore)
            fonescore[i].append(thef1score)
        
    return accscore, fonescore, std, importances, tprs, aucs

def visualizeFeat(tfeats, allnames):
    #@tfeats: a list of feature rankings generated from the boruta package

    mfeats = np.mean(tfeats,axis=0)
    mstds = np.std(tfeats,axis=0)
    

    indices = np.argsort(mfeats)
    temp = np.array(allnames)[indices]
    mfeats = np.sort(mfeats)
    oneindices = np.where(mfeats < 2)[0]
    notoneindices = np.where(mfeats > 2)[0]


    #Features on y-axis
    #plt.yticks(np.arange(0,len(temp)*2,step=1), temp)
    plt.xticks(np.arange(0,30,step=1))
    plt.errorbar(x=mfeats[oneindices], y=temp[oneindices], xerr=mstds[indices][oneindices], fmt='bo',ecolor='magenta')
    plt.errorbar(x=mfeats[notoneindices[0:20]], y=temp[notoneindices[0:20]], xerr=mstds[indices][notoneindices[0:20]], fmt='r+',ecolor='green')

    #Must go after, otherwise it does not work, and labels overlap with each other
    plt.tight_layout()
    plt.title("Feature Importance (Lower is Better)")
    plt.ylabel("Feature Name")
    plt.xlabel("Importance")


    #plt.show()

    return indices

def visualizeFeat2(imp_mean, imp_med, imp_rank):
    '''
        @tfeats: a list of feature rankings generated from the boruta package
        References:  
        PairGrid: https://seaborn.pydata.org/examples/pairgrid_dotplot.html
        Saving: https://stackoverflow.com/questions/32244753/how-to-save-a-seaborn-plot-into-a-file
        Tight Layout: https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
    '''

    import pandas as pd
    # Convert numpy_array of arrays into dataframe, column names are the features    
    meandf = pd.DataFrame(data=imp_mean, columns=f2names)

    # Convert to long-form format for sea-born where column is Features and Values, and the each feature within the feature column is described it's name
    meandf_long = meandf.melt(var_name="Features")

    # Calculate means of all the features to get sorted index, so that we can order the seaborn plot (HAD TO DO THIS BECAUSE SEABORN ORDERS ON DATAFRAME,
    # IF WE PASS IN A SORTED DATAFRAME IN POINTPLOT IT ORDERS BY RESULTING PANDAS DATAFRAME) (Basically if we get rid of order argument in sns.pointplot it will not look sorted)
    all_mean_df = meandf_long.groupby(["Features"]).mean().reset_index()
    ordered = all_mean_df.sort_values(by="value",ascending=False)["Features"]

    sns.set(style="whitegrid")

    # Creats a grid with values
    g = sns.PairGrid(meandf_long.sort_values(by="value", axis=0,ascending=False).reset_index(),
                    x_vars=["value"], y_vars=["Features"], height=20)

    # Creates the PointPlot, only showing ordered[:num] Features
    # https://github.com/mwaskom/seaborn/blob/master/seaborn/palettes.py - s = start, r=rotation, h=hue (1_r = reverse of rotation)
    g.map(sns.pointplot, size=10, orient="h", order=ordered[:-20],
        palette="ch:s=4,r=-.1,h=1_r", linewidth=1, join=False, ci=95)
    g.set(xlabel="Importance", ylabel="Features")

    # Shows y-axis grid 
    g.axes.flat[0].set(title="Feature Importance")
    g.axes.flat[0].xaxis.grid(False)
    g.axes.flat[0].yaxis.grid(True)



    fig = g.fig
    fig.savefig("new_ivim_testing_feat_final_2.svg",bbox_inches='tight')

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



def visualizeResults2(accscore, fonescore, tprs, aucs, allnames, thetitle):
    '''@accscore is the accuracy score of the individual classifiers
        @fonescore is the F1 score of the individual classifiers
        @std standrad deviation of feature importances **CURRENTLY NOT IMPLEMENTED**
        @importances importances of classifiers that can return feature importances, only index 0 and 2 are currently implemetned
        where importances[0] is SVM RFE Linear and importances[2] is Random Forest Importance'''

    print ("Calculating F1")

    # Average F1 Score for every classifier
    #avg_f1 = [np.mean(scores) for scores in fonescore]
    #avg_acc = [np.mean(scores) for scores in fonescore]
    #avg_auc = [np.mean(scores) for scores in fonescore]
    print (sns.utils.ci(aucs[-1],which=95))
    print (sns.utils.ci(fonescore[-1],which=95))
    print (sns.utils.ci(accscore[-1],which=95))
    sns.set()
    f1 = plt.figure(1)
    ax = sns.boxplot(data=fonescore,palette="vlag",showfliers=False)
    #sns.swarmplot(data=fonescore, color=".2", alpha=0.3)
    ax.set(xticklabels=c_names)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_title('F1 Score of all Classifiers')
    print ("Accuracy")
    f2 = plt.figure(2)
    ax2 = sns.boxplot(data=accscore,palette="vlag",showfliers=False)
    #sns.swarmplot(data=accscore, color=".2", alpha=0.3)
    ax2.set(xticklabels=c_names)
    ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
    ax2.set_title('Accuracy of all Classifiers')
    print ("AUC")
    f3 = plt.figure(3)
    ax3 = sns.boxplot(data=aucs,palette="vlag",showfliers=False)
    #sns.swarmplot(data=aucs, color=".2", alpha=0.3)
    ax3.set(xticklabels=c_names)
    ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
    ax3.set_title('Area Under Curve of all Classifiers')

    f1.savefig("F1_Classifier_test2.svg" ,bbox_inches='tight')
    f2.savefig("Acc_Classifier_test2.svg",bbox_inches='tight')
    f3.savefig("AUC_Classifier_test2.svg",bbox_inches='tight')

    #f4 = plt.figure(4)
    #ax4 = sns.lineplot(data=tprs, hue="region", style="event", markers=True, dashes=False)
    
    #plt.show()

def visualizeHists(hist40, hist60, hist80, hist100, hist120):
    '''
    Args:
        hist#: Tuple of (tprs, aucs, mean_fpr, accscore, fonescore)
    '''

    hist_names = ["40 Bins", "60 Bins", "80 Bins", "100 bins", "120 bins"]
    sns.set()
    f1 = plt.figure(1)
    fonescore = [hist40[4], hist60[4], hist80[4], hist100[4], hist120[4]]
    ax = sns.boxplot(data=fonescore,palette="vlag")
    sns.swarmplot(data=fonescore, color=".2")
    ax.set(xticklabels=hist_names)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    ax.set_title('F1 Score of all Histograms')


    f2 = plt.figure(2)
    accscore = [hist40[3], hist60[3], hist80[3], hist100[3], hist120[3]]
    ax2 = sns.boxplot(data=accscore,palette="vlag")
    sns.swarmplot(data=accscore, color=".2")
    ax2.set(xticklabels=hist_names)
    ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45)
    ax2.set_title('Accuracy of all Histograms')

    f3 = plt.figure(3)
    aucs = [hist40[1], hist60[1], hist80[1], hist100[1], hist120[1]]
    ax3 = sns.boxplot(data=aucs,palette="vlag")
    sns.swarmplot(data=aucs, color=".2")
    ax3.set(xticklabels=hist_names)
    ax3.set_xticklabels(ax3.get_xticklabels(),rotation=45)
    ax3.set_title('Area Under Curve of all Histograms')

    f1.savefig("F1_Histogram.svg",bbox_inches='tight')
    f2.savefig("Acc_Histogram.svg",bbox_inches='tight')
    f3.savefig("AUC_Histogram.svg",bbox_inches='tight')

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

def runScript():

    #Get all the Files
    print ("Getting Files containing features of original maps")
    file_path = ['ctrw_newIvim_maxminFeat_80', 'new_maxminFeatIvim_80']
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

    print
    print ("Testing Original Data")

    atitle = "Original Data"
    accscore, fonescore, std, importances, tprs, aucs = runClassifiers(xtrain, ytrain)
    #visualizeResults(accscore, fonescore, std, importances, fnames, atitle)


    print
    print ("Testing Improvement of Augmented Data")

    #Random Shuffle of Augmentations

    randindx = np.random.choice(np.arange(len(full_aug_xtrain)),size=int(len(full_aug_xtrain)*0.5),replace=True)
    #xtrain = np.vstack((xtrain,full_aug_xtrain[randindx]))
    #ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain[randindx]))

    #What if we include all augmented data ???
    #xtrain = np.vstack((xtrain,full_aug_xtrain))
    #ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain))
    
    #Run Classifier for Augumented Data
    atitle = "Augumented Data"

    #accscore, fonescore, std, importances = runClassifiers(xtrain, ytrain)
    #visualizeResults(accscore, fonescore, std, importances, fnames, atitle)

    '''
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


    '''

    
visualizeResults
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

    #xtrain = np.vstack((xtrain,full_aug_xtrain[randindx]))
    #ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain[randindx]))
    
    #What if we train on all augmented data ???
    xtrain = np.vstack((xtrain,full_aug_xtrain))
    ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain))

    #Save full IVIM+CTRW dataset
    np.save("newivim_xtrain_full_ivim_ctrw.npy",xtrain)
    np.save("newivim_ytrain_full_ivim_ctrw.npy",ytrain)
    
    atitle = "Org and IVIM Maps"
    accscore, fonescore, std, importances, tprs, aucs = runClassifiers(xtrain, ytrain)
    #visualizeResults(accscore, fonescore, std, importances, f2names, atitle)

    '''
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


    # ### Pipeline for GridSearch 


    # #****** GRID SEARCH FOR OPTIMAL PARAMATERS for Gradient Boost Classifier   ********
    # print ("GRID SEARCH")

    # """ tuned_parameters = {"loss": ["deviance"], "learning_rate": [0.01, .025, .05, .075, 0.1, 0.15, 0.2], "min_samples_split": np.linspace(0.1,0.5,8),"min_samples_leaf": np.linspace(0.1,0.5,8),
    # "max_depth": [3,5,8],"max_features":["log2", "sqrt"],"criterion": ["friedman_mse"],"subsample": [0.5, 0.618, 0.8, 0.85, 1.0], "n_estimators":[100,250,500]}

    #grid_classifier = GradientBoostingClassifier()
    #classifier_name = names[-1]
    #the_estimator, after_tuned = runGridSearch(tuned_parameters,grid_classifier, xtrain, ytrain, f2names, classifer_name):

    #*******************
    '''

    after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': 'sqrt', 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
    'criterion': 'friedman_mse', 'n_estimators': 100}

    model = GradientBoostingClassifier(**after_tuned)
    tprs, aucs, mean_fpr, accscore, fonescore, model, b_feats = oneModel(model, xtrain, ytrain)
    #visRocCurve(tprs, aucs, mean_fpr, accscore, fonescore, model, f2names, b_feats)
    '''
    modulename = 'boruta' 
    if modulename not in sys.modules:
        #print ('You have not imported the {} module for feature selection').format(modulename)
        print ("package not installed, install BorutaPy")
    
    else:
        rkf2 = RepeatedKFold(n_splits=5, n_repeats=100) 
        bro_feats = []
        bro_imp_mean = []
        bro_imp_med = []
        i = 0
        print ("Feature Importance")
        #after_tuned = {'max_depth': 8, 'n_estimators': 100}
        for train, _ in rkf2.split(xtrain):
            borutamodel = GradientBoostingClassifier(**after_tuned)
            model_feat = BorutaPy(borutamodel,perc=80,max_iter=500,verbose=2)
            model_feat.fit(xtrain[train], ytrain[train].ravel())
            bro_feats.append(model_feat.ranking_)
            tmp_mean, tmp_median = model_feat.check_imp_history()
            bro_imp_mean.append(tmp_mean)
            bro_imp_med.append(tmp_median)
            i+=1
            print ("Total Iteration: {}".format(i))
             
        #visualize best features from boruta package
        file_name = 'best_feats_list_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_feats)
        file_name = 'best_feats_imp_mean_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_imp_mean)
        file_name = 'best_feats_imp_med_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_imp_med)
        #visualizeFeat(bro_feats, f2names)

        #print (model_feat.support_weak_)
    
        
    #temp = np.load('best_feats_list_0.npy')
    #b_feats2 = visualizeFeat(temp, f2names)

    #visRocCurve(tprs, aucs, mean_fpr, accscore, fonescore, model, f2names, b_feats)

    #tprs, aucs, mean_fpr, accscore, fonescore, model, b_feats2 = oneModel(model, xtrain[:,b_feats2[:10]], ytrain)

    #visRocCurve(tprs, aucs, mean_fpr, accscore, fonescore, model, f2names, b_feats2)
    '''
    print ("COMPLETED WOO")

def runScript2():
    ''' Load big training matrix and do analysis '''

    xtrain = np.load("xtrain_full_ivim_ctrw.npy")
    ytrain = np.load("ytrain_full_ivim_ctrw.npy")

    accscore, fonescore, std, importances = runClassifiers2(xtrain, ytrain)
    #visualizeResults(accscore, fonescore, std, importances, f2names, "All Data")



def runScript3():
    ''' Load big training matrix and do analysis '''

    # top 8 features
    temp2 = np.array([False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False, False,  True,
       False, False,  True,  True, False, False, False, False, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False,  True, False, False, False,  True, False])
        
    xtrain = np.load("xtrain_full_ivim_ctrw.npy")
    ytrain = np.load("ytrain_full_ivim_ctrw.npy")

    randindx = np.random.choice(np.arange(len(xtrain)),size=int(len(xtrain)*0.8),replace=False)

    cvxtrain = xtrain[randindx, :]
    cvxtrain = cvxtrain[:, temp2]
    cvytrain = ytrain[randindx, :]


    accscore, fonescore, std, importances, tprs, aucs = runClassifiers(cvxtrain, cvytrain)

    checkpoint = [accscore, fonescore, tprs, aucs]

    np.save("checkpoint_2_sepearte_test", checkpoint)

    #visualizeResults(accscore, fonescore, std, importances, f2names, "All Data")
    #visualizeResults2(accscore, fonescore, tprs, aucs, f2names, "All Data")

def runScript4():

    # Get Saved Results, mainly the ROC curve
    thedata = np.load("checkpoint_2_sepearte_test.npy")
    tprs = np.asarray(thedata[2])
    tprs_mean = np.asarray([np.mean(tpr,axis=0) for tpr in tprs])
    tprs_std = np.asarray([np.std(tpr, axis=0) for tpr in tprs])
    tprs_upper = tprs_mean + 2*tprs_std/(np.sqrt(50))
    tprs_lower = tprs_mean - 2*tprs_std/(np.sqrt(50))
    #err_traces, xs, _ = make_errors2(tprs_mean[0], (tprs_lower[0], tprs_upper[0]))
    #make_errors3(tprs_mean, (tprs_lower, tprs_upper),title="ROC Comparison of Classifiers", l_names=c_names)
    #layout=dict(title="First Plot", xaxis={'title':'x1'}, yaxis={'title':'x2'})

    #viz._send({'data': err_traces, 'layout': layout, 'win': 'mywin'})
    #viz._send({'data': err_traces2, 'layout': layout, 'win': 'mywin'})
    


def runScript5():

    # Get Saved Results, mainly the ROC curve
    thedata = np.load("checkpoint_2_sepearte_test.npy")
    accscore, fonescore, tprs, aucs = thedata
    accscore = [temp for temp in accscore]
    fonescore = [temp for temp in fonescore]
    tprs = [temp for temp in tprs]
    aucs = [temp for temp in aucs]
    print (np.median(aucs[-1]))
    print (sns.utils.ci(aucs[-1],which=95))
    #visualizeResults2(accscore, fonescore, tprs, aucs, f2names, "All Data")

def runScriptBro():
    after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': 'sqrt', 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
    'criterion': 'friedman_mse', 'n_estimators': 100}

    model = GradientBoostingClassifier(**after_tuned)
    #xtrain = np.load("xtrain_full_ivim_ctrw.npy")
    #ytrain = np.load("ytrain_full_ivim_ctrw.npy")
    xtrain = np.load("newivim_xtrain_full_ivim_ctrw.npy")
    ytrain = np.load("newivim_ytrain_full_ivim_ctrw.npy")
    tprs, aucs, mean_fpr, accscore, fonescore, model, b_feats = oneModel(model, xtrain, ytrain)
    
    modulename = 'boruta' 
    if modulename not in sys.modules:
        #print ('You have not imported the {} module for feature selection').format(modulename)
        print ("package not installed, install BorutaPy")
    
    else:
        rkf2 = RepeatedKFold(n_splits=5, n_repeats=50) 
        bro_feats = []
        bro_imp_mean = []
        bro_imp_med = []
        i = 0
        print ("Feature Importance")
        #after_tuned = {'max_depth': 8, 'n_estimators': 100}
        for train, _ in rkf2.split(xtrain):
            borutamodel = GradientBoostingClassifier(**after_tuned)
            model_feat = BorutaPy(borutamodel,perc=80,max_iter=20,verbose=2)
            model_feat.fit(xtrain[train], ytrain[train].ravel())
            bro_feats.append(model_feat.ranking_)
            tmp_mean, tmp_median = model_feat.check_imp_history()
            bro_imp_mean.append(tmp_mean)
            bro_imp_med.append(tmp_median)
            i+=1
            print ("Total Iteration: {}".format(i))
             
        #visualize best features from boruta package
        file_name = 'ivim_best_feats_list_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_feats)
        file_name = 'ivim_best_feats_imp_mean_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_imp_mean)
        file_name = 'best_feats_imp_med_{}.npy'.format(np.random.randint(0,200))
        np.save(file_name,bro_imp_med)

def runScriptVisBro():
    ''' Visualize Boruta Package Feature Importance Results
    '''

    # load feature importances

    bro_imp_mean = np.load('best_feats_imp_mean_88.npy')
    bro_imp_med = np.load('best_feats_imp_med_149.npy')
    bro_imp_rank = np.load('best_feats_list_168.npy')

    visualizeFeat2(bro_imp_mean, bro_imp_med, bro_imp_rank)


def runScriptHist40(model=[]):

    print ("Getting Files containing features of original maps")
    file_path = ['maxminFeat_40', 'maxminFeatIvim_40']
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

    xtrain = np.vstack((xtrain,full_aug_xtrain))
    ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain))
    
    tprs, aucs, mean_fpr, accscore, fonescore, model, b_feats = oneModel(model, xtrain, ytrain)

    return (tprs, aucs, mean_fpr, accscore, fonescore)

def runScriptHist60(model=[]):

    print ("Getting Files containing features of original maps")
    file_path = ['maxminFeat_60', 'maxminFeatIvim_60']
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

    xtrain = np.vstack((xtrain,full_aug_xtrain))
    ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain))

    tprs, aucs, mean_fpr, accscore, fonescore, model, b_feats = oneModel(model, xtrain, ytrain)

    return (tprs, aucs, mean_fpr, accscore, fonescore)

def runScriptHist100(model=[]):

    print ("Getting Files containing features of original maps")
    file_path = ['maxminFeat_100', 'maxminFeatIvim_100']
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

    xtrain = np.vstack((xtrain,full_aug_xtrain))
    ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain))

    tprs, aucs, mean_fpr, accscore, fonescore, model, b_feats = oneModel(model, xtrain, ytrain)

    return (tprs, aucs, mean_fpr, accscore, fonescore)


def runScriptHist120(model=[]):

    print ("Getting Files containing features of original maps")
    file_path = ['maxminFeat_120', 'maxminFeatIvim_120']
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

    xtrain = np.vstack((xtrain,full_aug_xtrain))
    ytrain = np.vstack((ytrain.reshape(-1,1),full_aug_ytrain))

    after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': 'sqrt', 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
    'criterion': 'friedman_mse', 'n_estimators': 100}

    tprs, aucs, mean_fpr, accscore, fonescore, model, b_feats = oneModel(model, xtrain, ytrain)

    return (tprs, aucs, mean_fpr, accscore, fonescore)


def runAllHists(model, model_name):

    #returns (tprs, aucs, mean_fpr, accscore, fonescore)
    

    hist_names = ["40 Bins", "60 Bins", "80 Bins", "100 bins", "120 bins"]
    print ("40")
    hist40 = runScriptHist40(model)
    print ("60")
    hist60 = runScriptHist60(model)

    # Chosed Histogram 80 bins
    xtrain = np.load("xtrain_full_ivim_ctrw.npy")
    ytrain = np.load("ytrain_full_ivim_ctrw.npy")

    tprs, aucs, mean_fpr, accscore, fonescore, _, _ = oneModel(model, xtrain, ytrain)
    ####
    hist80 = [tprs, aucs, mean_fpr, accscore, fonescore]
    print ("100")
    hist100 = runScriptHist100(model)
    print ("120")
    hist120 = runScriptHist120(model)


    # Create ROC plot comparing Histogram ROCS
    tprs = [hist40[0], hist60[0], hist80[0], hist100[0], hist120[0]]
    tprs_mean = np.asarray([np.mean(tpr,axis=0) for tpr in tprs])
    tprs_std = np.asarray([np.std(tpr, axis=0) for tpr in tprs])
    tprs_upper = tprs_mean + 2*tprs_std/(np.sqrt(50))
    tprs_lower = tprs_mean - 2*tprs_std/(np.sqrt(50))

    #make_errors3make_errors3(tprs_mean, (tprs_lower, tprs_upper), title="ROC Comparison of Histograms for {} classifer".format(model_name), l_names=hist_names)

    #visualizeHists(hist40, hist60, hist80, hist100, hist120, title)

def runMorefeats():

    alltprs=[]
        
    feat_names = ["15% of features", "25% of features", "35% of features", "50% of features"]
    #xtrain = np.load("xtrain_full_ivim_ctrw.npy")
    #ytrain = np.load("ytrain_full_ivim_ctrw.npy")

    xtrain = np.load("newivim_xtrain_full_ivim_ctrw.npy")
    ytrain = np.load("newivim_ytrain_full_ivim_ctrw.npy")
    after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': .15, 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
    'criterion': 'friedman_mse', 'n_estimators': 100}
    model = GradientBoostingClassifier(**after_tuned)
    tprs, aucs, mean_fpr, accscore, fonescore, _, _ = oneModel(model, xtrain, ytrain)
    alltprs.append(tprs)

    after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': .25, 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
    'criterion': 'friedman_mse', 'n_estimators': 100}
    model = GradientBoostingClassifier(**after_tuned)

    tprs, aucs, mean_fpr, accscore, fonescore, _, _ = oneModel(model, xtrain, ytrain)
    alltprs.append(tprs)

    after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': 0.35, 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
    'criterion': 'friedman_mse', 'n_estimators': 100}
    model = GradientBoostingClassifier(**after_tuned)

    tprs, aucs, mean_fpr, accscore, fonescore, _, _ = oneModel(model, xtrain, ytrain)
    alltprs.append(tprs)

    after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': 0.5, 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
    'criterion': 'friedman_mse', 'n_estimators': 100}
    model = GradientBoostingClassifier(**after_tuned)

    tprs, aucs, mean_fpr, accscore, fonescore, _, _ = oneModel(model, xtrain, ytrain)
    alltprs.append(tprs)

    
    tprs_mean = np.asarray([np.mean(tpr,axis=0) for tpr in alltprs])
    tprs_std = np.asarray([np.std(tpr, axis=0) for tpr in alltprs])
    tprs_upper = tprs_mean + 2*tprs_std/(np.sqrt(50))
    tprs_lower = tprs_mean - 2*tprs_std/(np.sqrt(50))

    #make_errors3(tprs_mean, (tprs_lower, tprs_upper), title="ROC Comparison of Number of Features", l_names=feat_names)


def runSpecificFeats():

    
    num_samples=10000
    alltprs=[]

    # top 8 features
    temp2 = np.array([False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False, False,  True,
       False, False,  True,  True, False, False, False, False, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False,  True, False, False, False,  True, False])
        
    feat_names = ["15% of features", "25% of features", "35% of features", "50% of features"]
    #xtrain = np.load("xtrain_full_ivim_ctrw.npy")
    #ytrain = np.load("ytrain_full_ivim_ctrw.npy")

    xtrain = np.load("newivim_xtrain_full_ivim_ctrw.npy")
    ytrain = np.load("newivim_ytrain_full_ivim_ctrw.npy")

    randindx = np.random.choice(np.arange(len(xtrain)),size=int(len(xtrain)*0.8),replace=False)

    cvxtrain = xtrain[randindx, :]
    cvxtrain = cvxtrain[:, temp2]
    cvytrain = ytrain[randindx, :]

    xtest = np.delete(xtrain,randindx,axis=0)
    xtest = xtest[:, temp2]
    ytest = np.delete(ytrain,randindx,axis=0)

    after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': None, 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
    'criterion': 'friedman_mse', 'n_estimators': 100}
    model = GradientBoostingClassifier(**after_tuned)

    model.fit(cvxtrain,cvytrain.ravel())
    allf1 = np.empty((len(classifiers),0)).tolist()
    allauc = np.empty((len(classifiers),0)).tolist()
    allacc = np.empty((len(classifiers),0)).tolist()
    print ("F1, Accuracy, AUC")
    ##############3 Bootstrap
    print ("GB")
    testacc = []
    testauc = []
    testf1 = []
    for i in range(num_samples):
        # prepare train and test sets
        indicies = resample(np.arange(0,40), n_samples=30, replace=False)
        # evaluate model
        ypred = model.predict(xtest[indicies,:])
        probz = model.predict_proba(xtest[indicies,:])
        fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
        testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
        testauc.append(auc(fpr,tpr))
        testf1.append(f1_score(ytest[indicies].ravel(),ypred))

    allf1[-1].append(testf1)
    allauc[-1].append(testauc)
    allacc[-1].append(testacc)

    print (np.std(testf1), np.std(testacc), np.std(testauc))   
    print (np.mean(testf1), np.mean(testacc), np.mean(testauc))
    print (np.median(testf1), np.median(testacc), np.median(testauc)) 
    model = classifiers[0]
    print (c_names[0])
    model.fit(cvxtrain,cvytrain.ravel())
    testacc = []
    testauc = []
    testf1 = []

    for i in range(num_samples):
        # prepare train and test sets
        indicies = resample(np.arange(0,40), n_samples=30, replace=False)
        # evaluate model
        ypred = model.predict(xtest[indicies,:])
        probz = model.predict_proba(xtest[indicies,:])
        fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
        testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
        testauc.append(auc(fpr,tpr))
        testf1.append(f1_score(ytest[indicies].ravel(),ypred))

    print (np.std(testf1), np.std(testacc), np.std(testauc))   
    print (np.mean(testf1), np.mean(testacc), np.mean(testauc))
    print (np.median(testf1), np.median(testacc), np.median(testauc))
    allf1[0].append(testf1)
    allauc[0].append(testauc)
    allacc[0].append(testacc)

    model = classifiers[1]
    print (c_names[1])
    model.fit(cvxtrain,cvytrain.ravel())
    testacc = []
    testauc = []
    testf1 = []
    for i in range(num_samples):
        # prepare train and test sets
        indicies = resample(np.arange(0,40), n_samples=30, replace=False)
        # evaluate model
        ypred = model.predict(xtest[indicies,:])
        probz = model.predict_proba(xtest[indicies,:])
        fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
        testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
        testauc.append(auc(fpr,tpr))
        testf1.append(f1_score(ytest[indicies].ravel(),ypred))

    print (np.std(testf1), np.std(testacc), np.std(testauc))   
    print (np.mean(testf1), np.mean(testacc), np.mean(testauc)) 
    print (np.median(testf1), np.median(testacc), np.median(testauc))

    allf1[1].append(testf1)
    allauc[1].append(testauc)
    allacc[1].append(testacc)


    model = classifiers[2]
    print (c_names[2])
    model.fit(cvxtrain,cvytrain.ravel())
    testacc = []
    testauc = []
    testf1 = []
    for i in range(num_samples):
        # prepare train and test sets
        indicies = resample(np.arange(0,40), n_samples=30, replace=False)
        # evaluate model
        ypred = model.predict(xtest[indicies,:])
        probz = model.predict_proba(xtest[indicies,:])
        fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
        testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
        testauc.append(auc(fpr,tpr))
        testf1.append(f1_score(ytest[indicies].ravel(),ypred))

    print (np.std(testf1), np.std(testacc), np.std(testauc))   
    print (np.mean(testf1), np.mean(testacc), np.mean(testauc)) 
    print (np.median(testf1), np.median(testacc), np.median(testauc))
    allf1[2].append(testf1)
    allauc[2].append(testauc)
    allacc[2].append(testacc)

    model = classifiers[3]
    print (c_names[3])
    model.fit(cvxtrain,cvytrain.ravel())
    testacc = []
    testauc = []
    testf1 = []
    for i in range(num_samples):
        # prepare train and test sets
        indicies = resample(np.arange(0,40), n_samples=30, replace=False)
        # evaluate model
        ypred = model.predict(xtest[indicies,:])
        probz = model.predict_proba(xtest[indicies,:])
        fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
        testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
        testauc.append(auc(fpr,tpr))
        testf1.append(f1_score(ytest[indicies].ravel(),ypred))

    print (np.std(testf1), np.std(testacc), np.std(testauc))   
    print (np.mean(testf1), np.mean(testacc), np.mean(testauc))
    print (np.median(testf1), np.median(testacc), np.median(testauc)) 
    allf1[3].append(testf1)
    allauc[3].append(testauc)
    allacc[3].append(testacc)

    model = classifiers[4]
    print (c_names[4])
    model.fit(cvxtrain,cvytrain.ravel())
    testacc = []
    testauc = []
    testf1 = []
    for i in range(num_samples):
        # prepare train and test sets
        indicies = resample(np.arange(0,40), n_samples=30, replace=False)
        # evaluate model
        ypred = model.predict(xtest[indicies,:])
        probz = model.predict_proba(xtest[indicies,:])
        fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
        testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
        testauc.append(auc(fpr,tpr))
        testf1.append(f1_score(ytest[indicies].ravel(),ypred))

    print (np.std(testf1), np.std(testacc), np.std(testauc))   
    print (np.mean(testf1), np.mean(testacc), np.mean(testauc)) 
    print (np.median(testf1), np.median(testacc), np.median(testauc))
    allf1[4].append(testf1)
    allauc[4].append(testauc)
    allacc[4].append(testacc)

    model = classifiers[5]
    print (c_names[5])
    model.fit(cvxtrain,cvytrain.ravel())
    testacc = []
    testauc = []
    testf1 = []
    for i in range(num_samples):
        # prepare train and test sets
        indicies = resample(np.arange(0,40), n_samples=30, replace=False)
        # evaluate model
        ypred = model.predict(xtest[indicies,:])
        probz = model.predict_proba(xtest[indicies,:])
        fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
        testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
        testauc.append(auc(fpr,tpr))
        testf1.append(f1_score(ytest[indicies].ravel(),ypred))

    print (np.std(testf1), np.std(testacc), np.std(testauc))   
    print (np.mean(testf1), np.mean(testacc), np.mean(testauc)) 
    print (np.median(testf1), np.median(testacc), np.median(testauc))
    allf1[5].append(testf1)
    allauc[5].append(testauc)
    allacc[5].append(testacc)

    print (c_names[6])
    model = classifiers[6]
    model.fit(cvxtrain,cvytrain.ravel())
    testacc = []
    testauc = []
    testf1 = []
    for i in range(num_samples):
        # prepare train and test sets
        indicies = resample(np.arange(0,40), n_samples=30, replace=False)
        # evaluate model
        ypred = model.predict(xtest[indicies,:])
        probz = model.predict_proba(xtest[indicies,:])
        fpr, tpr, thresholds = roc_curve(ytest[indicies].ravel(),probz[:,1])
        testacc.append(accuracy_score(ytest[indicies].ravel(), ypred))
        testauc.append(auc(fpr,tpr))
        testf1.append(f1_score(ytest[indicies].ravel(),ypred))
    
    allf1[6].append(testf1)
    allauc[6].append(testauc)
    allacc[6].append(testacc)

    np.save('test_results_new_ivim', [allf1, allauc, allacc])

    print (np.std(testf1), np.std(testacc), np.std(testauc))   
    print (np.mean(testf1), np.mean(testacc), np.mean(testauc)) 
    print (np.median(testf1), np.median(testacc), np.median(testauc))

    #visualizeResults2(allacc, allf1, [], allauc, [], [])

	#ax2 = sns.boxplot(data=[testf1, testacc, testauc], palette="vlag",showfliers=False )
    #ax2.set_xticklabels(["F1 Score", "Accuracy", "Area Under the Curve"])
    #ax2.set_title("Gradient Boosted Classifier Test Set")
    #plt.show()

    # Get Saved Results, mainly the ROC curve
    #thedata = np.load("checkpoint_2_sepearte_test.npy")
    #tprs = np.asarray(thedata[2])
    #tprs_mean = np.asarray([np.mean(tpr,axis=0) for tpr in tprs])
    #tprs_std = np.asarray([np.std(tpr, axis=0) for tpr in tprs])
    #tprs_upper = tprs_mean + 2*tprs_std/(np.sqrt(50))
    #tprs_lower = tprs_mean - 2*tprs_std/(np.sqrt(50))
    #err_traces, xs, _ = make_errors2(tprs_mean[0], (tprs_lower[0], tprs_upper[0]))
    #make_errors4(tprs_mean, (tprs_lower, tprs_upper),opt_tpr=the_tpr, title="ROC Comparison of Classifiers", l_names=c_names)

    #print (thef1score, theaccscore, auccore)
    #print (themodel.feature_importances_)
    
    #alltprs.append(tpr)

def runTemp():

    allf1, allauc, allacc = np.load('test_results3.npy')
    allf1 = allf1[:,0,:]
    allauc = allauc[:,0,:]
    allacc = allacc[:,0,:]

    accscore = [temp for temp in allacc]
    fonescore = [temp for temp in allf1]
    aucs = [temp for temp in allauc]

    visualizeResults2(accscore, fonescore, [], aucs, [], [])


#runScript()

#runScript3()

#runScript4()

#runScript5()

#runScriptBro()

#runScriptVisBro()

#after_tuned = {'min_samples_leaf': 0.15714285714285714, 'learning_rate': 0.2, 'loss': 'deviance', 'max_features': 'sqrt', 'subsample': 0.8, 'max_depth': 8, 'min_samples_split': 0.1, 
#'criterion': 'friedman_mse', 'n_estimators': 100}
#model = GradientBoostingClassifier(**after_tuned)s

#for i in list(range(1,6)):
#    runAllHists(classifiers[i], c_names[i])

#runAllHists(model, c_names[7])

#runMorefeats()

#runSpecificFeats()

#runTemp()

""" # Generate random classification example: https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data
from sklearn.datasets import make_blobs
# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.tight_layout()
plt.savefig('random_classification.svg')
print ("JELLO") """