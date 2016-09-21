# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:44:10 2016
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
"""
# Local
from automannerplus import *


alignpath = '/Users/itanveer/Data/ROCSpeak_BL/features/alignments/'
timepath = '/Users/itanveer/Data/ROCSpeak_BL/Original_Data/Results/'
timepath_turk = '/Users/itanveer/Data/ROCSpeak_BL/Original_Data/Results_for_MTurk/'
gtfile = '/Users/itanveer/Data/ROCSpeak_BL/Ground-Truth/participants_ratings.csv'
gtfile_turk = '/Users/itanveer/Data/ROCSpeak_BL/Ground-Truth/turkers_ratings.csv'
prosodypath = '/Users/itanveer/Data/ROCSpeak_BL/features/prosody/'
transpath = '/Users/itanveer/Data/ROCSpeak_BL/Ground-Truth/Transcripts/'

# Main
if __name__=='__main__':
    # Generates the data file (Run for the first time)
    # ================================================
    utils.save_for_classify(alignpath,timepath,gtfile,prosodypath)
    utils.save_for_classify_mtark(alignpath,timepath_turk,gtfile_turk,prosodypath)
    utils.save_all_features(alignpath,timepath,timepath_turk,gtfile,gtfile_turk,prosodypath,transpath)
    
    # ------------- Misc Stats ------------
    utils.calc_misc_stat(alignpath,timepath,timepath_turk,gtfile,gtfile_turk,prosodypath,transpath)


    ######################################################################
    # Uses the data file for classification 
    # (Run if the data file is generated already)
    # =====================================
    
    # ------------ Regression -------------
    Use the mechanical turk annotations
    cls1 = Classify('all_features_MT_gt.pkl')
    for item in cls1.featnames:
        print item
    cls1.test_avg_corr(show_plots=True,tot_iter=1000,method='lasso')
    cls1.test_avg_corr(show_plots=True,tot_iter=1000,method='lda')
    cls1.test_avg_corr(show_plots=True,tot_iter=1000,method='max-margin') 
    # Use the participants' self annotations
    cls2 = Classify('all_features_gt.pkl')
    cls2.test_avg_corr(show_plots=True,tot_iter=1000,method='lasso')    
    cls2.test_avg_corr(show_plots=True,tot_iter=1000,method='lda')
    cls2.test_avg_corr(show_plots=True,tot_iter=1000,method='max-margin')
    
    # Perform t-test to see if there is any significant difference between
    # two groups of accuracies or correlation coefficients
    print 'p value (lasso-MTS regression vs lasso-self regression):',\
        utils.perform_t_test(cls1,'lasso','regression',True,\
            cls2,'lasso','regression',False,100)
    print 'p value (lda-MTS regression vs lda-self regression):',\
        utils.perform_t_test(cls1,'lda','regression',True,\
            cls2,'lda','regression',False,100)
    print 'p value (SVR-MTS regression vs SVR-self regression):',\
        utils.perform_t_test(cls1,'max-margin','regression',True,\
            cls2,'max-margin','regression',False,100)
    print 'p value (lasso-MTF regression vs lasso-self regression):',\
        utils.perform_t_test(cls1,'lasso','regression',False,\
            cls2,'lasso','regression',False,100)
    print 'p value (lda-MTF regression vs lda-self regression):',\
        utils.perform_t_test(cls1,'lda','regression',False,\
            cls2,'lda','regression',False,100)
    print 'p value (SVR-MTF regression vs SVR-self regression):',\
        utils.perform_t_test(cls1,'max-margin','regression',False,\
            cls2,'max-margin','regression',False,100)
    
    # ------------ Classification -------------
    cls1 = Classify('all_features_MT_gt.pkl')
    cls1.test_avg_corr(show_plots=True,tot_iter=1000,task = 'classification',method='lda')    
    cls1.test_avg_corr(show_plots=False,method='max-margin',task='classification',tot_iter=10)
    # Use the participants' self annotations
    cls2 = Classify('all_features_gt.pkl')
    cls2.test_avg_corr(show_plots=True,method='max-margin',task='classification',tot_iter=10)    
    cls2.test_avg_corr(show_plots=True,tot_iter=1000,task = 'classification',method='lda')

    Perform t-test to see if there is any significant difference between
    two groups of accuracies or correlation coefficients
    print 'p value (lda-MTS classification vs lda-self classification):',\
        utils.perform_t_test(cls1,'lda','classification',True,\
            cls2,'lda','classification',False,100)
    print 'p value (SVR-MTS classification vs SVR-self classification):',\
        utils.perform_t_test(cls1,'max-margin','classification',True,\
            cls2,'max-margin','classification',False,100)
    print 'p value (lda-MTF classification vs lda-self classification):',\
        utils.perform_t_test(cls1,'lda','classification',False,\
            cls2,'lda','classification',False,100)
    print 'p value (SVR-MTF classification vs SVR-self classification):',\
        utils.perform_t_test(cls1,'max-margin','classification',False,\
            cls2,'max-margin','classification',False,100)
            
    # ----------- Classification with LASSO ---------
    cls1 = Classify_MLP('./all_features_MT_gt.pkl')
    cls1.test_avg_corr(show_all=True,task='classification',method='LASSO')
    cls2 = Classify_MLP('./all_features_gt.pkl')
    cls2.test_avg_corr(show_all=True,task='classification',method='LASSO')

    # Perform t-test
    t1 = utils.perform_t_test(cls1,'LASSO','classification',True,\
            cls2,'LASSO','classification',False,10)
    t2 = utils.perform_t_test(cls1,'LASSO','classification',False,\
            cls2,'LASSO','classification',False,10)
    import pdb;pdb.set_trace()
    print 'p value (lasso-MTS classification vs lasso-self classification):',t1    
    print 'p value (lasso-MTF classification vs lasso-self classification):',t2
        

    # ----------- Classification with MLP ---------
    cls1 = Classify_MLP('./all_features_MT_gt.pkl')
    # cls1.test_avg_corr(task='classification')
    cls2 = Classify_MLP('./all_features_gt.pkl')
    # cls2.test_avg_corr(task='classification')
    t1 = utils.perform_t_test(cls1,'mlp','classification',True,\
            cls2,'mlp','classification',False,30)
    t2 = utils.perform_t_test(cls1,'mlp','classification',False,\
            cls2,'mlp','classification',False,10)
    import pdb;pdb.set_trace()
    print 'p value (MLP-MTS classification vs MLP-self classification):',t1
    print 'p value (MLP-MTF classification vs MLP-self classification):',t2

    # ----------- Regression with MLP ---------
    cls1 = Classify_MLP('./all_features_MT_gt.pkl')
    cls1.test_avg_corr()
    cls2 = Classify_MLP('./all_features_gt.pkl')
    cls2.test_avg_corr()

    # Perform t-test
    t1 = utils.perform_t_test(cls1,'mlp','regression',True,\
            cls2,'mlp','regression',False,100)
    t2 = utils.perform_t_test(cls1,'mlp','regression',False,\
            cls2,'mlp','regression',False,100)
    import pdb;pdb.set_trace()
    print 'p value (MLP-MTS regression vs MLP-self regression):',t1
    print 'p value (MLP-MTF regression vs MLP-self regression):',t2