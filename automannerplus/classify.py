"""
Created on Tue Jun 28 3:19:10 2016
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
"""

# Python lib
import numpy as np
from scipy.stats import expon
import cPickle as cp
import time


# Plot related
import matplotlib.pyplot as plt

# ML related
import sklearn as sk
from sklearn.grid_search import *
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

'''
A class to apply classifiers and obtain the accuracy and other metrics

Class variables:
=============== 
x        : Features
y        : Labels
data     : All contents of the pkl file
totfeat  : Total number of features
featnames: Name of the features
'''
class Classify(object):
    def __init__(self,pklfilename):
        self.filename = pklfilename
        data = cp.load(open(pklfilename,'rb'))
        # Extract from data
        self.__X__ = [[float(item) for item in dat] for vid in data['X'].keys()\
            for dat in data['X'][vid]]
        self.y = [item for vid in data['Y'].keys() for item in data['Y'][vid]]
        # Total number of features
        self.totfeat = np.size(self.__X__,axis=1)
        self.featnames = data['featurename']
        # Use all features
        self.usefeat()

    # Turn on/off a specific group of features
    # and standardize the features
    def usefeat(self,disf=True,pros=True,body=True,face=True,lex=True):
        self.disf=disf
        self.pros=pros
        self.body=body  
        self.face=face
        if self.totfeat==123:
            self.lex=lex
        # Standardize the features, x
        self.x = np.nan_to_num(self.__X__/np.std(self.__X__,axis=0))
        self.x = self.x - np.mean(self.x,axis=0)
        # Select the features according to the mask
        x_ = np.empty(0).reshape(len(self.x),0)
        if self.disf:
            x_ = np.hstack((x_,self.x[:,:9]))
        if self.pros:
            x_ = np.hstack((x_,self.x[:,9:35]))
        if self.body:
            x_ = np.hstack((x_,self.x[:,35:76]))
        if self.face:
            x_ = np.hstack((x_,self.x[:,76:100]))
        if self.lex:
            x_ = np.hstack((x_,self.x[:,100:123]))
        self.x = x_

    # Test avg. correlation for multiple regressions
    def test_avg_corr(self,
        show_all=False,
        show_plots=False,
        method='lasso', # Method of classification
        task='regression', # Task can be regression or classification
        tot_iter = 30,  # Total number of repeated experiment
        paramtuning=True,
        ):
        if task=='regression':
            # Train and test the classifier many times for calculating the accuracy
            correl = []
            coefs = []
            for i in xrange(tot_iter):
                if show_all:
                    print 'iter:',i,
                # One third of the data is reserved for testing
                x_train,x_test,y_train,y_test = \
                    sk.cross_validation.train_test_split(\
                    self.x,self.y,test_size=0.3,random_state=\
                    int(time.time()*1000)%4294967295)
                # Model Selection: use LASSO
                if method=='lasso':
                    model = linear_model.Lasso(alpha=0.05,\
                        fit_intercept=True, \
                        normalize=False, precompute=False, copy_X=True, \
                        max_iter=1000000, tol=0.0001, warm_start=False, \
                        positive=False,selection='random')
                    # Training the model
                    model.fit(x_train,y_train)
                    if self.disf and self.pros and self.body and self.face and self.lex:
                        modelcoef = model.coef_
                        coefs.append(self.__coef_calc__(modelcoef))
                # LDA for regression
                elif method=='lda':
                    model = sk.discriminant_analysis.\
                        LinearDiscriminantAnalysis(
                        solver='lsqr',
                        shrinkage='auto')
                    # Training the model
                    model.fit(x_train,y_train)
                    if self.disf and self.pros and self.body and self.face and self.lex:
                        modelcoef = model.coef_
                        coefs.append(self.__coef_calc__(np.mean(\
                            np.abs(modelcoef),axis=0)))
                # Max margin for regression
                elif method=='max-margin':
                    model = sk.svm.LinearSVR(
                        C = 0.01,fit_intercept=True,random_state=\
                        int(time.time()*1000)%4294967295)
                    model.fit(x_train,y_train)
                    if self.disf and self.pros and self.body and self.face and self.lex:
                        modelcoef = model.coef_
                        coefs.append(self.__coef_calc__(modelcoef))
                # Prediction results
                y_pred = model.predict(x_test)
                # Calculate correlation with original
                corr_val = np.corrcoef(y_test,y_pred)[0,1]
                correl.append(corr_val)
                if show_all:
                    print 'Correlation:',corr_val
        elif task=='classification':
            # Train and test the classifier many times for calculating the accuracy
            correl = []
            coefs = []
            fpr = []
            tpr = []
            # Labels for classification
            Y_ = 2.*(np.array(self.y)>3.0).astype(float)-1.

            # Iterate for averaging the accuracy
            for i in xrange(tot_iter):
                if paramtuning:
                    # Half of the data is reserved as Evaluation set
                    x_train,x_test,y_train,y_test = \
                        sk.cross_validation.train_test_split(\
                        self.x,Y_,test_size=0.5,random_state=\
                        int(time.time()*1000)%4294967295)  
                    param_grid = {'C':expon(loc=0.,scale=3)}
                    clf = RandomizedSearchCV(sk.svm.LinearSVC(penalty='l1',\
                        dual=False,fit_intercept=True),param_grid,cv=5,\
                        scoring='roc_auc',n_iter=50)
                    clf.fit(x_train,y_train)
                    print 'best param (C)',clf.best_params_['C']

                    if show_all:
                        print 'iter:',i,
                    # Only max-margin for classification
                    if method=='max-margin':
                        model = sk.svm.LinearSVC(C = clf.best_params_['C'],
                            penalty='l1',dual=False,fit_intercept=True,
                            random_state=int(time.time()*1000)%4294967295)
                        model.fit(x_train,y_train)
                        if self.disf and self.pros and self.body and self.face and self.lex:
                            if model.coef_.ndim>1:
                                modelcoef = model.coef_[0]
                            coefs.append(self.__coef_calc__(modelcoef))
                    else:
                        raise ValueError("Method "+method+" not supported yet")
                else:
                    # Train test split
                    x_train,x_test,y_train,y_test = \
                        sk.cross_validation.train_test_split(\
                        self.x,Y_,test_size=0.4,random_state=\
                        int(time.time()*1000)%4294967295)  
                    if show_all:
                        print 'iter:',i,
                    # Only max-margin for classification
                    if method=='max-margin':
                        model = sk.svm.LinearSVC(C = 0.05,
                            penalty='l1',dual=False,fit_intercept=True,
                            random_state=int(time.time()*1000)%4294967295)
                        model.fit(x_train,y_train)
                        if self.disf and self.pros and self.body and \
                            self.face and self.lex:
                            if model.coef_.ndim>1:
                                modelcoef = model.coef_[0]
                            coefs.append(self.__coef_calc__(modelcoef))
                    else:
                        raise ValueError("Method "+method+" not supported yet")

                # Prediction results
                y_pred = model.predict(x_test)
                y_score = model.decision_function(x_test)
                # Calculate correlation with original
                corr_val = sk.metrics.roc_auc_score(y_test,y_pred)
                fpr_temp,tpr_temp,_ = sk.metrics.roc_curve(y_test,y_score)
                
                tpr.append(np.interp(np.linspace(0,1,100),fpr_temp,tpr_temp))
                correl.append(corr_val)
                if show_all:
                    print 'Accuracy:',corr_val
            plt.figure()
            plt.plot(np.linspace(0,1,100),np.mean(tpr,axis=0),label='ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.savefig('Outputs/ROC_Curve_'+self.filename+'_'+\
                method+'_'+task+'.pdf',format='pdf')

        # Print feature proportions
        meancorrel = np.mean(correl)
        print '======================================'
        print 'Task:',task,'Method:',method
        if task=='regression':
            print 'Average correlation:', meancorrel
        else:
            print 'Average Accuracy:', meancorrel
        print '======================================'
        # Print grouped coefficient values
        if self.disf and self.pros and self.body and self.face and self.lex:
            coef = np.mean(coefs,axis=0)
            print 'disf:', coef[0]
            print 'pros:', coef[1]
            print 'body:', coef[2]
            print 'face:', coef[3]
            print 'lexical:',coef[4]
            print 'Number of disf feats:', coef[5]
            print 'Number of pros feats:', coef[6]
            print 'Number of body feats:', coef[7]
            print 'Number of face feats:', coef[8]
            print 'Number of lexical feats:', coef[9]
            print 'disf percent:',coef[-5]
            print 'prosody percent:',coef[-4]
            print 'Body Percent:', coef[-3]
            print 'Face Percent:',coef[-2]
            print 'Lexical Percent:',coef[-1]
            print '--------------------------------'

        if modelcoef.ndim>1:
            modelcoef = modelcoef[0,:]
        # Print all the sorted coef values
        srIdx = np.argsort(-np.abs(modelcoef))
        for i in range(self.totfeat):
            print self.featnames[srIdx[i]]+':',modelcoef[srIdx[i]]
        print           
        # Plot pie charts of the grouped coefficient values
        if self.disf and self.pros and self.body and self.face and self.lex:        
            # Visualize feature proportions
            plt.figure(figsize=(7,3))
            plt.pie(coef[-5:],autopct='%1.1f%%',
                labels=['disfluency','prosody','body_movements','face','lexical'],
                colors = ['royalblue', 'darkkhaki', 'lightskyblue',\
                 'lightcoral','yellowgreen'])
            plt.axis('equal')
            plt.subplots_adjust(top=0.75)
            plt.title('Coefficient Ratio: '+task+'_'+method, y=1.10)
            plt.savefig('Outputs/coef_ratio_'+self.filename+'_'+\
                method+'_'+task+'%0.2f' % meancorrel+'.pdf',format='pdf')
        if show_plots:
            plt.show()
    
    # Calculates the relative weights of the coefficients
    # 1. Total weights for disfluency, prosody and body features
    # 2. Number of non-zeros for ... 
    # 3. Weights per unit non-zero feature
    # 4. Percent of feature categories
    def __coef_calc__(self,coef):
        disf = coef[:9]    # Disfluency features
        pros = coef[9:35]  # Prosody features
        body = coef[35:76] # Body features
        face = coef[76:100]   # Face features
        if self.lex:
            lexic = coef[100:123] # lexical features

        # Total weights
        sum_disf = np.sum(np.abs(disf))
        sum_pros = np.sum(np.abs(pros))
        sum_body = np.sum(np.abs(body))
        sum_face = np.sum(np.abs(face))
        if self.lex:
            sum_lexic = np.sum(np.abs(lexic))
        # Number of features
        count_disf = len(disf)
        count_pros = len(pros)
        count_body = len(body)
        count_face = len(face)
        if self.lex:
            count_lexic = len(lexic)
        # ratios
        rat_disf = sum_disf/float(count_disf) if not count_disf==0 else 0.
        rat_pros = sum_pros/float(count_pros) if not count_pros==0 else 0.
        rat_body = sum_body/float(count_body) if not count_body==0 else 0.
        rat_face = sum_face/float(count_face) if not count_face==0 else 0.
        if not self.lex:
            total_rat = rat_disf + rat_pros + rat_body + rat_face
        else:
            rat_lexic = sum_lexic/float(count_lexic) if not count_lexic==0 else 0.
            total_rat = rat_disf + rat_pros + rat_body + rat_face + rat_lexic
        # percents
        perc_disf = rat_disf/total_rat * 100. if not total_rat==0 else 0.
        perc_pros = rat_pros/total_rat * 100. if not total_rat==0 else 0.
        perc_body = rat_body/total_rat * 100. if not total_rat==0 else 0.
        perc_face = rat_face/total_rat * 100. if not total_rat==0 else 0.
        if self.lex:
            perc_lexic = rat_lexic/total_rat * 100. if not total_rat==0 else 0.

        if not self.lex:
            return sum_disf,sum_pros,sum_body,sum_face,count_disf,\
            count_pros,count_body,count_face,rat_disf,rat_pros,\
            rat_body,rat_face,perc_disf,perc_pros,perc_body,perc_face
        else:
            return sum_disf,sum_pros,sum_body,sum_face,sum_lexic,count_disf,\
            count_pros,count_body,count_face,count_lexic,rat_disf,rat_pros,\
            rat_body,rat_face,rat_lexic,perc_disf,perc_pros,perc_body,\
            perc_face,perc_lexic
