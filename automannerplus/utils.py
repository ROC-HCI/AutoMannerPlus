"""
Created on Tue Jun 28 3:19:10 2016
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
"""

# Local
from krip_alpha import alpha, interval_metric
from automannerplus import AutoMannerPlus, AutoMannerPlus_mturk
from collections import defaultdict as ddict
# Python lib
import numpy as np
import cPickle as cp
# Plot related
import matplotlib.pyplot as plt

# Calculates the misc statistics
def calc_misc_stat(
    # Turker's ground truth
    alignpath,
    timepath,
    timepath_turk,
    gtfile,
    gtfile_turk,
    prosodypath,
    transpath,
    showplot = False,
    vids = ['34.1','35.2','36.1','37.2','38.1','39.2','40.1','41.2','42.1',
            '44.1','45.2','46.1','47.2','48.1','49.2','50.1','51.2','52.1',
            '53.2','54.1','55.2','56.1','57.2','58.1','59.2','60.1','61.2',
            '62.1']
    ):
    
    # ===================== Participants ============================
    amp = AutoMannerPlus(gtfile,alignpath,timepath,prosodypath)
    
    amp_m = AutoMannerPlus_mturk(gtfile_turk,alignpath,timepath_turk,prosodypath)
    # in the amp_m.gt, the 3 turker's ratings are averaged. But we want each
    # turker's individual rating. This is extracted in GT_Raw
    GT_Raw = amp_m.__readGT__(gtfile_turk,raw_gt=True)

    # Inter-rater Agreements among 3 turkers
    data = np.zeros((3,0))    
    for avid in vids:
        data = np.concatenate((data,GT_Raw[avid]),axis=1)  # Accumulating data
    
    # Calculating the stat
    ka1 = alpha(data,metric=interval_metric)    # Calculate Krippendorff's Alpha
    cor1 = np.sum(np.triu(np.corrcoef(data),1))/3  # Calculate average correlation coefficient

    print 'Agreement among turkers',ka1,'corrcoef',cor1
    # ----------------------------------------------------
    data = np.zeros((2,0))
    for avid in vids:
        temp = np.vstack((amp.gt[avid],GT_Raw[avid][0,:]))
        data = np.concatenate((data,temp),axis=1)
    ka2 = alpha(data,metric=interval_metric)

    cor2 = np.corrcoef(data)[0,1]
    print 'Agreement between first turker and participant',ka2,'corrcoef',cor2
    # ----------------------------------------------------
    data = np.zeros((2,0))
    for avid in vids:
        temp = np.vstack((amp.gt[avid],GT_Raw[avid][1,:]))
        data = np.concatenate((data,temp),axis=1)
    ka3 = alpha(data,metric=interval_metric)
    cor3 = np.corrcoef(data)[0,1]
    print 'Agreement between second turker and participant',ka3,'corrcoef',cor3
    # ----------------------------------------------------
    data = np.zeros((2,0))
    for avid in vids:
        temp = np.vstack((amp.gt[avid],GT_Raw[avid][2,:]))
        data = np.concatenate((data,temp),axis=1)
    ka4 = alpha(data,metric=interval_metric)
    cor4 = np.corrcoef(data)[0,1]
    print 'Agreement between third turker and participant',ka4,'corrcoef',cor4
    # ----------------------------------------------------
    data = np.zeros((2,0))
    for avid in vids:       
        temp = np.vstack((amp.gt[avid],np.mean(GT_Raw[avid],axis=0)))
        data = np.concatenate((data,temp),axis=1)
    ka5 = alpha(data,metric=interval_metric)
    cor5 = np.corrcoef(data)[0,1]
    print 'Agreement between mean-turker and participant',ka5,'corrcoef',cor5
    # ----------------------------------------------------
    
    ########### Draw Relative inter rater agreements ############
    opacity = 0.5
    plt.figure('Agreements')
    plt.bar([1,2,3,4,5],[ka1,ka2,ka3,ka4,ka5],0.3,alpha=opacity,\
        color = 'b',label='Krippendorrf\'s Alpha')
    plt.bar([1.3,2.3,3.3,4.3,5.3],[cor1,cor2,cor3,cor4,cor5],0.3,\
        alpha=opacity,color = 'r',label='Correlation Coefficient')
    plt.xticks([1,2,3,4,5],['Among Turkers', 'Turker-1 vs Participant',\
        'Turker-2 vs Participant','Turker-3 vs Participant',\
        'Turker-Average vs Participant'],rotation=40)
    plt.text(1,ka1+0.01,'%0.2f'%ka1)
    plt.text(2,ka2+0.01,'%0.2f'%ka2)
    plt.text(3,ka3+0.01,'%0.2f'%ka3)
    plt.text(4,ka4+0.01,'%0.2f'%ka4)
    plt.text(5,ka5+0.01,'%0.2f'%ka5)
    plt.text(1.3,cor1+0.01,'%0.2f'%cor1)
    plt.text(2.3,cor2+0.01,'%0.2f'%cor2)
    plt.text(3.3,cor3+0.01,'%0.2f'%cor3)
    plt.text(4.3,cor4+0.01,'%0.2f'%cor4)
    plt.text(5.3,cor5+0.01,'%0.2f'%cor5)
    plt.ylim([0,0.65])
    plt.ylabel('Agreement Scores')
    plt.subplots_adjust(bottom=0.4)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('Outputs/agreements.pdf',format='pdf')

    #################################################################
    # Mechanical Turk Average Rating Distribution
    bin_mean = count_ratings(amp_m.gt)
    # Mechanical Turk First Annotator Rating Distribution
    bin_first = count_ratings({key:val[0,:].tolist() for key,val in GT_Raw.items()})
    # Mechanical Turk Second Annotator Rating Distribution
    bin_second = count_ratings({key:val[1,:].tolist() for key,val in GT_Raw.items()})
    # Mechanical Turk Third Annotator Rating Distribution
    bin_third = count_ratings({key:val[2,:].tolist() for key,val in GT_Raw.items()})
    # Participants Rating Distribution
    bin_part = count_ratings(amp.gt)

    # Draw relative distribution of labels
    idx = np.arange(1,8)
    bar_width = 1./6.
    opacity = 0.5
    plt.figure('Label Distribution',figsize=(10,6))
    plt.plot(idx,bin_first,alpha=opacity,color='b',label='First Turker Annotation')
    plt.plot(idx,bin_second,alpha=opacity,color='r',label='Second Turker Annotation')
    plt.plot(idx,bin_third,alpha=opacity,color='g',label='Third Turker Annotation')
    plt.plot(idx,bin_mean,alpha=opacity,color='k',label='Mean Turker Annotation')
    plt.plot(idx,bin_part,alpha=opacity,color='darkkhaki',label='Participant\'s Annotation')
    plt.xlabel('Ratings')
    plt.ylabel('Label Counts')
    plt.xticks(idx,('1','2','3','4','5','6','7'))
    plt.legend()
    plt.savefig('Outputs/label_dist.pdf',format='pdf')
    if showplot:
        plt.show()

# Count the distributions of the annotations
def count_ratings(GT):
    dist_counts = {}
    for avid in GT.keys():
        for arate in GT[avid]:
            if arate==0:
                continue
            if not arate in dist_counts.keys():                
                dist_counts[arate] = 0
            else:
                dist_counts[arate] += 1
    # Return the distribution
    return [dist_counts[akey] if akey in dist_counts.keys() \
        else 0 for akey in range(1,8)]

# Process and Save all the features and participants' ground truth for classical classification
# This approach uses the data from the participants' ratings
def save_for_classify(
    alignpath,
    timepath,
    gtfile,
    prosodypath,
    outfilename = 'features_gt.pkl',
    vids = ['34.1','35.2','36.1','37.2','38.1','39.2','40.1','41.2','42.1',
            '44.1','45.2','46.1','47.2','48.1','49.2','50.1','51.2','52.1',
            '53.2','54.1','55.2','56.1','57.2','58.1','59.2','60.1','61.2',
            '62.1'] 
    ):
    amp = AutoMannerPlus(gtfile,alignpath,timepath,prosodypath)
    X_data = ddict(list)
    Y_data = ddict(list)
    for avid in vids:
        print 'processing ...',avid
        amp.readfast(avid)
        features,gt_ = amp.extractfeaturesfast()

        for i in features.keys():
            X_data[avid].append(features[i])
            Y_data[avid].append(gt_[i])

    print 'Dump all data to features_gt.pkl file'
    cp.dump({'X':X_data,'Y':Y_data,'featurename':amp.featurename()},open(outfilename,'wb'))

# Process and Save all the features and Mechanical Turker's ground truth for classical classification
# This approach uses the data from the participants' ratings
def save_for_classify_mtark(
    alignpath,
    timepath,
    gtfile,
    prosodypath,
    outfilename = 'features_MT_gt.pkl'
    ):
    amp = AutoMannerPlus_mturk(gtfile,alignpath,timepath,prosodypath)
    
    # for all the files, extract features
    X_data = ddict(list)
    Y_data = ddict(list)
    # File list can be found in gt dict
    for avid in amp.gt.keys():
        print 'processing ...',avid
        amp.readfast(avid)
        features,gt_ = amp.extractfeaturesfast()
        for i in features.keys():
            X_data[avid].append(features[i])
            Y_data[avid].append(gt_[i])
    print 'Dump all data to features_MT_gt.pkl file'
    cp.dump({'X':X_data,'Y':Y_data,'featurename':amp.featurename()},\
        open(outfilename,'wb'))

# Saves ALL possible features (for both participants and MTurk)
def save_all_features(
    alignpath,
    timepath,
    timepath_turk,
    gtfile,
    gtfile_turk,
    prosodypath,
    transpath,
    outfilename = 'all_features_MT_gt.pkl',
    vids = ['34.1','35.2','36.1','37.2','38.1','39.2','40.1','41.2','42.1',
            '44.1','45.2','46.1','47.2','48.1','49.2','50.1','51.2','52.1',
            '53.2','54.1','55.2','56.1','57.2','58.1','59.2','60.1','61.2',
            '62.1']
    ):

    # ===================== Participants ============================
    amp = AutoMannerPlus(gtfile,alignpath,timepath,prosodypath)
    
    X_data = ddict(list)
    Y_data = ddict(list)
    # Read all the features from every file 
    for avid in vids:
        print 'processing ...',avid
        amp.readEverything(transpath,avid)
        features,gt_ = amp.extractAllFeatures()
        # For every pattern
        for i in features.keys():
            X_data[avid].append(features[i])
            Y_data[avid].append(gt_[i])
    print 'Dump all data to all_features_gt.pkl file'
    # dump the features with appropriate filenames
    cp.dump({'X':X_data,'Y':Y_data,'featurename':\
        amp.featurename()},open('all_features_gt.pkl','wb'))

    # ======================== MTURK ================================
    # for all the files, extract features
    amp_m = AutoMannerPlus_mturk(gtfile_turk,alignpath,timepath_turk,prosodypath)
    X_data = ddict(list)
    Y_data = ddict(list)
    # File list can be found in gt dict
    for avid in amp_m.gt.keys():
        print 'processing ...',avid
        amp_m.readEverything(transpath,avid)
        features,gt_ = amp_m.extractAllFeatures()
        # For every pattern
        for i in features.keys():
            X_data[avid].append(features[i])
            Y_data[avid].append(gt_[i])
    print 'Dump all data to all_features_MT_gt.pkl file'
    cp.dump({'X':X_data,'Y':Y_data,'featurename':\
        amp_m.featurename(),'GT_full':amp_m.__gt_full__},\
        open(outfilename,'wb'))            