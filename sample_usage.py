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
    # utils.save_for_classify(alignpath,timepath,gtfile,prosodypath)
    # utils.save_for_classify_mtark(alignpath,timepath_turk,gtfile_turk,prosodypath)
    # utils.save_all_features(alignpath,timepath,timepath_turk,gtfile,gtfile_turk,prosodypath,transpath)
    # # ------------- Misc Stats ------------
    # utils.calc_misc_stat(alignpath,timepath,timepath_turk,gtfile,gtfile_turk,prosodypath,transpath)

    # Uses the data file for classification 
    # (Run if the data file is generated already)
    # =====================================
    # ------------ Regression -------------
    # Use the mechanical turk annotations
    cls = classify('all_features_MT_gt.pkl')
    cls.test_avg_corr(show_plots=False,tot_iter=1000,method='lasso')
    cls.test_avg_corr(show_plots=False,tot_iter=1000,method='lda')
    cls.test_avg_corr(show_plots=False,tot_iter=1000,method='max-margin')
    # Use the participants' self annotations
    cls = classify('all_features_gt.pkl')
    cls.test_avg_corr(show_plots=False,tot_iter=1000,method='lasso')    
    cls.test_avg_corr(show_plots=False,tot_iter=1000,method='lda')
    cls.test_avg_corr(show_plots=True,tot_iter=1000,method='max-margin')
    # ------------ Classification -------------
    cls = classify('all_features_MT_gt.pkl')
    cls.test_avg_corr(show_plots=False,method='max-margin',task='classification',tot_iter=10)
    # Use the participants' self annotations
    cls = classify('all_features_gt.pkl')
    cls.test_avg_corr(show_plots=True,method='max-margin',task='classification',tot_iter=10)
