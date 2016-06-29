
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
from LIWC_processor import match, ReadLIWCDictionary, ReadLIWCCategories

# Python lib
import numpy as np
import csv
from collections import defaultdict as ddict
from scipy.signal import resample
import os

# NLP
import nltk

''' 
    A class extracting features for classifying gestures as mannerism or meaningful.
    In order to extract the features, it aligns the transcript of the speech with the
    original speech sound file. This alignment disregards some words in the transcript
    as those are not there in the aligner's dictionary (so it doesn't know how to pronounce it)

    Class variables:
    =======================
    walign             : Numpy array of all the aligned words and their startime, endtime.
                         Its dtype=[('word', 'S15'), ('stime', '<f2'), ('etime', '<f2')].
                         Please note, the alignment word list skips some words because their
                         pronounciation pattern was not known. In addition, the alignment word
                         may contain additional 'sp' word which indicates silence or no voice.
    trans              : A list of all the transcription words. Note that the transcription words
                         are tokenized and lowercase. So "I've" is broken into "i" and "'ve".
    pos                : A List of all the POS tags for the corresponding transcription words.
    align2trmap        : a dictionary from alignment word index (key) to transcription word (value)
                         index. If an alignment word is not found in trans list (e.g. sp) then the
                         value is set to -1. Note, this operation performs a dynamic programming based
                         alignment operation which can become time-consuming
    gt                 : A dictionary from vidid (string) to the selected ground truth data.
    patterns           : A numpy array of (sorted) pattern# and its start time and end time (in sec).
                       : It represents all the time locations where a specific pattern is occurred.
    vidid              : id (string) of the current video
    lnkdata            : A dictionary from pattern id to the words spoken within that pattern.
    lnkdata_pos        : Similar to lnkdata but contains POS
    alignpath, 
    timepath,      
    trasnpath          : various pathnames
    selected           : A dictionary from (pat,inst) tuple to indices of walign
                         that falls within that inst's time period.
    sorted2unsorted    : A dictionary from sorted to unsorted pattern ID's
    facedat            : A dictionary from the pattern# to the corresponding
                         face data
    bodydat            : A dict from pattern# to body movement features
    liwc_categories    : List of LIWC category ID's used as features
    LIWCDic            : A dictionary from word to corresponding category ID's
    categories         : A dictionary from LIWC category ID to category name
    featurenames       : A list of all the features (grows with additional file read)
'''
class AutoMannerPlus(object):
    # Provide the filenames: pattern-timeline file, aligned transcript file
    def __init__(self,
        gtfilename, # ground truth filename
        alignpath,  # path where alignfile resides
        timepath,   # path where timeline file resides (provide the folder
                    # containing the video name folder)
        prosodypath # path where the prosody files are located
        ):
        self.__readGT__(gtfilename)
        self.alignpath = alignpath
        self.timepath = timepath
        self.prosodypath = prosodypath
        self.jointlist = [5,6,9,10,13,14,17,18]
        self.filler = ['uhh', 'um', 'uh', 'umm', 'like','say','basically',\
        'so', 'ah', 'umhum', 'uhum', 'uhm','oh','oho','oooh','huh','ah','aha']
        self.liwc_categories = [2,4,5,8,10,11,16,17,18,20,21,22,126,127,128,129,130,131,\
            137,140,250,354,463,464]
        path = os.path.abspath(__file__)
        curdir = os.path.dirname(path)
        self.LIWCDic = ReadLIWCDictionary(os.path.join(curdir,'data','liwcdic2007.dic'))
        self.categories = ReadLIWCCategories(os.path.join(curdir,'data','liwccat2007.txt'))
        # Add feature names
        self.featurenames = ['Average time to speak word','Average time to speak filler word',\
            'Average pause length','Total number of words','Total number of filler words',\
            'Total number of pauses','% of word per instance','% of filler per instance',\
            '% of pause per instance','Mean loudness','Minimum loudness','Maximum loudness',\
            'loudness range','loudness standard deviation','Mean pitch','Minimum pitch',\
            'Maximum pitch','pitch range','pitch standard deviation','Minimum first formant',\
            'Minimum second formant','Minimum third formant','Maximum first formant',\
            'Maximum second formant','Maximum third formant','Mean first formant',\
            'Mean second formant','Mean third formant','first formant std','second formant std',\
            'third formant std','first formant range','second formant range',\
            'third formant range','percent unvoiced']

    # A convenience function to view the ground truth data
    def viewgt(self):
        patlist = np.unique(self.patterns[:,0])
        print 'patternID','\t','old_patternID','\t','GT_Score'
        for idx,apat in enumerate(patlist):
            print apat,'\t','\t',self.sorted2unsorted[apat],\
                '\t','\t',self.gt[self.vidid][idx]

    # A method for getting the ground truth for ith pattern
    # provide a video id and a pattern number (i) for that video
    def getgt(self,vidid,i):
            return self.gt[vidid][np.where(np.unique(self.patterns[:,0])==i)[0][0]] 

    # A method to return the names of all the active features
    def featurename(self):
        return self.featurenames      
        
    # Extract features from the aligned transcript data
    # Features extracted in this function are as follows (per instance):
    # mean_wrdtime, mean_filltime, mean_pausetime, #w, #fill, #pause,
    # word%_inst, filler%_inst, pause%_inst & the prosody features
    def extractfeaturesfast(self):
        def id(pat,inst):
            return np.where(self.patterns[:,0]==pat)[0][inst]
        featurelist = ddict(list)
        gtlist={}
        # for every pattern i and instance j
        for i,j in self.selected.keys():
            if self.getgt(self.vidid,i)==0.:
                continue
            # instance start and end time
            inststime = float(self.patterns[id(i,j)][1])
            instetime = float(self.patterns[id(i,j)][2])
            # length of instance
            instLen = instetime - inststime
            # Length of each word (except filler)
            wrdTime = [self.walign['etime'][item] - self.walign['stime'][item] for item in\
                self.selected[i,j] if (not self.walign['word'][item]=='sp')\
                and (not self.walign['word'][item] in self.filler)]
            # Length of each filler word
            fillTime =[self.walign['etime'][item] - self.walign['stime'][item] for item in\
                self.selected[i,j] if self.walign['word'][item] in self.filler]
            # Length of each pause
            spTime =[self.walign['etime'][item] - self.walign['stime'][item] for item in\
                self.selected[i,j] if self.walign['word'][item]=='sp']
            # Prosody Features
            loud = self.loud[int(inststime*100.):int(instetime*100.)]
            pitch = self.pitch[int(inststime*100.):int(instetime*100.)]
            formant = self.formant[int(inststime*100.):int(instetime*100.),:]
            form_min = np.min(formant,axis=0)
            form_max = np.max(formant,axis=0)
            form_mean = np.mean(formant,axis=0)
            form_std = np.std(formant,axis=0)
            form_range = np.max(formant,axis=0) - np.min(formant,axis=0)
            # Summing the features for corresponding patterns (to calculate avg)
            feats = [\
                        # ================== Disfluency features (9) ===================
                        np.mean(wrdTime)if wrdTime else 0.0, # Average time to speak words
                        np.mean(fillTime)if fillTime else 0.0, # Average time to speak filler words
                        np.mean(spTime) if spTime else 0.0,    # Average pause length
                        len(wrdTime) if wrdTime else 0.0,      # Total number of words
                        len(fillTime) if fillTime else 0.0,    # Total number of filler words
                        len(spTime) if spTime else 0.0,        # Total number of pauses
                        np.sum(wrdTime)/instLen if wrdTime else 0.0, # % of word per instance
                        np.sum(fillTime)/instLen if fillTime else 0.0, # % of filler per instance
                        np.sum(spTime)/instLen if spTime else 0.0, # % of filler per instance
                        # ==================== Prosody Features (26) =====================
                        np.mean(loud),                     # Mean loudness
                        np.min(loud),                      # Minimum loudness
                        np.max(loud),                      # Maximum loudness
                        np.max(loud) - np.min(loud),       # loudness range
                        np.std(loud),                      # loudness standard deviation
                        np.mean(pitch),                    # Mean pitch
                        np.min(pitch),                     # Minimum pitch
                        np.max(pitch),                     # Maximum pitch
                        np.max(pitch) - np.min(pitch),     # pitch range
                        np.std(pitch),                     # pitch standard deviation
                        form_min[0],                       # Minimum first formant
                        form_min[1],                       # Minimum second formant
                        form_min[2],                       # Minimum third formant
                        form_max[0],                       # Maximum first formant
                        form_max[1],                       # Maximum second formant
                        form_max[2],                       # Maximum third formant
                        form_mean[0],                      # Mean first formant
                        form_mean[1],                      # Mean second formant
                        form_mean[2],                      # Mean third formant
                        form_std[0],                       # first formant std
                        form_std[1],                       # second formant std
                        form_std[2],                       # third formant std
                        form_range[0],                     # first formant range
                        form_range[1],                     # second formant range
                        form_range[2],                     # third formant range
                        np.count_nonzero(pitch)/len(pitch) # percent unvoiced
                        ]
            # Add features
            featurelist[i].append(feats)
        for i in np.unique(self.patterns[:,0]):
             # ================= feature averaging ===================
            if not len(featurelist[i])==0:
                featurelist[i] = np.mean(featurelist[i],axis=0).tolist()
                # =============== Body Features (40) =====================
                featurelist[i].extend(self.bodydat[i])
                # =============== Facial Features (24) ===================
                featurelist[i].extend(self.facedat[i])
                # Ground truth
                gtlist[i]=self.getgt(self.vidid,i)
            else:
                del featurelist[i]
        return featurelist,gtlist

    # Extract lexical features
    def __lexical_feature__(self,patid):
        feat_ = {akey:0. for akey in self.liwc_categories}
        # If the lexical feature is not available for this pattern
        if not patid in self.lnkdata.keys():
            return [feat_[k] for k in feat_.keys()]
        # If lexical feature is available
        wrdlist = [words for sublist in self.lnkdata[patid] for words in sublist]
        # Check wordlist and count the liwc categories as features
        for aword in wrdlist:
            for acat in match(self.LIWCDic,aword):
                if acat in self.liwc_categories:
                    feat_[acat]+=1.
        return [feat_[k] for k in feat_.keys()]

    # This is the full version of feature extraction. call it only if you 
    # used the readEverything (not readfast) function earlier.
    # It calculates lexical features in addition to the original features
    def extractAllFeatures(self):
        if not self.fullfeat:
            raise ValueError(\
                "Can't extract all features unless everything is read. Use readEverything.")
        # It includes all the previous features
        featlist,gtlist = self.extractfeaturesfast()
        # Extract lexical features for each pattern
        for i in featlist.keys():
            feat_ = self.__lexical_feature__(i)
            featlist[i].extend(feat_)
        # Append lexical feature names
        featname = ['count_'+self.categories[item] for item in self.liwc_categories]
        self.featurenames.extend(featname)
        return featlist,gtlist
        
    # View the length of the spoken words
    def viewwordlen(self):
        # for every instances in the patterns
        for i,j in self.selected.keys():
            print i,j,'=',[(self.walign['word'][item],self.walign['etime'][item]-\
            self.walign['stime'][item]) for item in self.selected[i,j]]

    # Read files in full detail.
    def readEverything(self,transpath,vidid):
        self.transpath = transpath
        self.vidid = vidid
        # Read the files
        self.__read_align__(self.alignpath+vidid+'.txt')
        self.__read_time__(self.timepath+vidid+'/timeline_'+vidid+'.csv')
        self.__read_prosody__()
        self.__read__body_movements__()
        self.__selectwalign__()
        self.__read_facial__()
        # Read original transcript and Build the links with transcript data. 
        # This part is time consuming
        self.__read_trans__(self.transpath+vidid+'.txt')        
        self.__buildalign2trmap__()
        self.__lnwordpatt__()
        # Indicate as full feature
        self.fullfeat = True
        
    # Faster read without the transcription data. Some features don't need 
    # transcripts. However, transcript data requires a time-consuming 
    # alignment process which could be avoided in a faster group of feature
    # extraction. Please note, this function doesn't prepare the variables
    # align2trmap, lnkdata, lnkdata_pos. So, after calling this
    # function, those variables are either unavailable or non-updated
    def readfast(self,vidid):
        self.vidid = vidid
        # Read the files
        self.__read_align__(self.alignpath+vidid+'.txt')
        
        self.__read_time__(self.timepath+vidid+'/timeline_'+vidid+'.csv')
        self.__read_prosody__()
        self.__read__body_movements__()
        self.__selectwalign__()
        self.__read_facial__()
        # Indicate as partial feature
        self.fullfeat = False

    # Reads the ground truth data files and makes the global gt variable
    # the fieldname variable contains the name of the columns that we are interested in
    # PID contains the indices for the column titled "This body movement pattern conveys a meaning"
    def __readGT__(self,
        filename,
        fieldname='This body movement pattern conveys a meaning.'
        ):
        alldata = []
        with open(filename,'r') as f:
            # reading the header and calculating column id (pid) for meaningfulness rating
            header = f.readline().split(',')[1:]
            pid = [idx for idx,item in enumerate(header) if item==fieldname]
            # Reading actual data
            for arow in csv.reader(f):
                alldata.append([item if i==0 else float(item) if item else 0. \
                    for i,item in enumerate(arow[1:])])
            # Make dict
            self.gt = {item[0]:[item[idx] for idx in pid] for item in alldata}

    # Read and extract facial features
    def __read_facial__(self):
        # Video Frame Rate is 29.97
        FPS = 29.97
        # facial features
        face_path = self.prosodypath.replace('prosody','facial')+\
            self.vidid+'.mp4.csv'
        with open(face_path,'r') as f:
            header = f.readline().strip().split(',')
            data=[]
            # read each line from file
            for aline in f:
                line = list()
                for item in aline.strip().split(',')[:-1]:
                    if item!='':
                        line.append(float(item))
                    else:
                        break
                if len(line)>2:
                    data.append(line)
                else:
                    continue
            data = np.array(data)
            if len(data)>0:
                data[:,0] /= FPS    # Convert the frame number to time
        # Pattern-wise assignment
        self.facedat = {}
        # for a pattern
        for i in np.unique(self.patterns[:,0]):
            patlist = self.patterns[self.patterns[:,0]==i]
            # Bypass empty data
            if len(data)==0:
                self.facedat[i] = np.zeros(24)
                continue
            # for a time-instance of the pattern
            feats = []
            for j in patlist:
                # frame-indices of this time-instance
                idx = np.where(np.bitwise_and(data[:,0]>=j[1], \
                    data[:,0]<=j[2]))[0]
                # Bypass empty data
                if len(idx)==0:
                    continue
                # Mean and variance of facial features
                face_feat = np.mean(data[idx,2:14],axis=0).tolist()
                face_feat.extend(np.std(data[idx,2:14],axis=0).tolist())
                feats.append(face_feat)
            # Average of features for each instance
            if len(feats)!=0:
                self.facedat[i] = np.mean(feats,axis=0).tolist()
            else:
                self.facedat[i] = np.zeros(24)
        # Add feature names
        feat_name=[]
        feat_name.extend(['mean_'+ahead for ahead in header[2:14]])
        feat_name.extend(['std_'+ahead for ahead in header[2:14]])
        self.featurenames.extend(feat_name)
                
    # Read the prosody files
    def __read_prosody__(self):
        # Read loudness
        loudfile = self.prosodypath + self.vidid + '.loud'
        pitchfile = self.prosodypath + self.vidid + '.pitch'
        formantfile = self.prosodypath + self.vidid + '.formant'
        self.loud = []
        with open(loudfile) as f:
            for aline in f:
                dat = aline.strip()
                if dat.startswith('z [1] ['):
                    self.loud.append(float(dat.split('=')[1]))
        self.loud = np.array(self.loud,dtype='float')
        # Read pitch
        nextline = False
        pitchcount = -1
        with open(pitchfile) as f:
            for idx,aline in enumerate(f):
                if idx == 5:
                    m = int(aline.strip().split('=')[1].strip())
                    self.pitch = np.zeros(m,dtype='float')                    
                if nextline:
                    self.pitch[pitchcount] = float(aline.strip().split('=')[1])
                    nextline = False
                if aline.strip()=='candidate [1]:':
                    nextline = True
                # Count the pitch entries
                if aline.startswith('    frame ['):
                    pitchcount += 1
        # Read formant
        nextline1 = False
        nextline2 = False
        nextline3 = False
        formcount = -1
        with open(formantfile) as f:
            for idx,aline in enumerate(f):
                # Reading the length in sec
                if idx == 4:
                    timemax = float(aline.strip().split('=')[1].strip())
                # reading the total file length
                if idx == 5:
                    m = int(aline.strip().split('=')[1].strip())
                    self.formant = np.zeros((m,3),dtype='float')
                # acting based on state (saving the formants)
                if nextline1:
                    self.formant[formcount,0]=float(aline.strip().split('=')[1])
                    nextline1=False
                if nextline2:
                    self.formant[formcount,1]=float(aline.strip().split('=')[1])
                    nextline2=False
                if nextline3:
                    self.formant[formcount,2]=float(aline.strip().split('=')[1])
                    nextline3=False
                # Setting the correct state (checking if the formant)
                if aline.strip()=='formant [1]:':
                    nextline1=True
                if aline.strip()=='formant [2]:':
                    nextline2=True
                if aline.strip()=='formant [3]:':
                    nextline3=True
                # Counting the formant stack
                if aline.startswith('    frames ['):
                    formcount += 1
        self.formant = resample(self.formant,int(timemax*100))
        
    # Read the aligned transcript file
    def __read_align__(self,filename):
        with open(filename) as f:
            alldata = f.readlines()
        if not alldata[0].strip()=='File type = "ooTextFile short"':
            raise ValueError('Not a correct alignment file')
        # Go upto the word tier
        idx_s = [i+5 for i,item in enumerate(alldata) if item.strip()== \
        '"IntervalTier"' and alldata[i+1].strip()=='"word"'][0]
        # parse and fill
        self.walign = np.array([(alldata[i+2].strip().strip('"').lower(),\
                float(alldata[i].strip()),float(alldata[i+1].strip())) for i in \
                range(idx_s,len(alldata),3)],\
                dtype=[('word','a15'),('stime','f2'),('etime','f2')])
    
    # Read the original transcript file
    def __read_trans__(self,transfile):
        with open(transfile,'r') as f:
            txt = f.read().decode('utf-8')
        self.trans = nltk.word_tokenize(txt.lower())
        self.pos = [item[1] for item in nltk.pos_tag(self.trans)]
                
    # Read a timeline data: patternID, startsec,endsec
    # It relabels the patterns according to the order
    # of the highest frequency
    def __read_time__(self,filename):
        with open(filename,'r') as f:
            f.readline()
            self.patterns = np.array([[rows[0]]+rows[2:].strip().split(',')[1:]\
                for rows in f],dtype=int)
            patdat_cpy = self.patterns[:,0].copy()
            # Relabeling the patterns
            freq = np.bincount(self.patterns[:,0])
            srti = np.argsort(-freq,kind='mergesort')
            for i,item in enumerate(srti):
                patdat_cpy[self.patterns[:,0]==item]=i
            # filling the sorted2unsorted variable
            self.sorted2unsorted = {item1:item2 for item1,item2\
                in zip(patdat_cpy,self.patterns[:,0])}
            # Relabeling the original
            self.patterns[:,0]=patdat_cpy
            
    # Method for calculating the body movement features.
    def __read__body_movements__(self):
        # Get Skeleton file path
        filepath = self.prosodypath.replace(\
            'features','allSkeletons').replace('prosody/',self.vidid+'.csv')
        with open(filepath,'r') as f:
            head = f.readline().strip().split(',')
            # Read all data and filter the unnecessary columns
            data = np.array([[float(item) if item else 0. for item\
                in x.strip().split(',')] for x in f])
        cols = np.array(list(set(range(102))-set(range(5,102,5))-\
            set(range(6,102,5))))
        data = data[:,cols]
        head = [head[item] for item in cols[2:]]
        # Calculate features
        self.bodydat = {}
        # for a pattern
        for i in np.unique(self.patterns[:,0]):
            patlist = self.patterns[self.patterns[:,0]==i]
            feats = []
            # for a time-instance of the pattern
            for j in patlist:
                # frame-indices of this time-instance
                idx = np.where(np.bitwise_and(data[:,1]/1000.>=j[1], \
                    data[:,1]/1000.<=j[2]))[0]
                # Bypass empty data
                if len(idx)==0:
                    continue
                # Calculate body movement features
                feats.append(self.__calcbodyfeat__(data[idx,2:]))
            # Average of instance features for each pattern
            if len(feats)!=0:
                self.bodydat[i] = np.mean(feats,axis=0).tolist()
            else:
                self.bodydat[i] = np.zeros(40)
        # Add feature-names
        feat_list = []
        for j in self.jointlist:
            feat_list.extend(['mean_vel_'+head[3*j][:-2],
            'mean_acc_'+head[3*j][:-2],
            'std_pos_'+head[3*j][:-2],
            'std_vel_'+head[3*j][:-2],
            'std_acc_'+head[3*j][:-2]])
        self.featurenames.extend(feat_list)

    # Calculate the body movement features
    def __calcbodyfeat__(self,data):
        # jid = joint id
        def jid(j):
            return np.arange(3*j,3*j+3)
        def length(v):
            return np.sqrt(v[:,0]**2.+v[:,1]**2.+v[:,2]**2.)
        features = []
        # Calculate features for only some selected joints
        for j in self.jointlist:
            # represent joints wrt the reference joint (hip)
            pos = data[:,jid(j)]-data[:,jid(0)]
            # normalize length to reduce person dependency
            pos = pos/np.mean(length(pos))
            # calculate the velocity and accelerations
            vel = np.diff(pos,axis=0)
            acc = np.diff(pos,n=2,axis=0)
            # Calculate features
            features.append(np.mean(length(vel)))     # Mean joint velocity mag
            features.append(np.mean(length(acc)))     # Mean joint acceleration mag
            features.append(np.std(length(pos)))      # STD joint position mag
            features.append(np.std(length(vel)))      # STD joint velocity mag
            features.append(np.std(length(acc)))      # STD joint acceleration mag
        return features

    # Create a map frm the alignment word to the transcript word (ignore sp)
    # using dynamic programming. This is important because the aligned 
    # transcript file has missing words. Also, the sentence boundary and 
    # other POS information is relative to the original transcript.   
    def __buildalign2trmap__(self):
        # Forced alignment wordlist
        alist = [item for item in self.walign['word'].tolist()]
        # Original wordlist
        blist = self.trans
        # initialization
        d = np.zeros((len(alist)+1,len(blist)+1))
        bp = np.zeros((len(alist)+1,len(blist)+1),dtype='i2,i2')
        if not (alist and blist):
            raise ValueError('Atleast one list is empty')
        d[:,0]=np.arange(np.size(d,axis=0))
        d[0,:]=np.arange(np.size(d,axis=1))        
        # Build up the distance and backpointer tables
        for i in range(1,len(alist)+1):
            for j in range(1,len(blist)+1):
                choices = [d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+2 \
                                if not alist[i-1]==blist[j-1] else d[i-1,j-1]]
                temp = np.argmin(choices)
                d[i,j] = choices[temp]
                bp[i,j] = [(i-1,j),(i,j-1),(i-1,j-1)][temp]
        # Build up the alignment from alist to blist
        self.align2trmap = {idx:-1 for idx in range(len(alist))}
        nd = (-1,-1)    
        while not (nd[0]==0 and nd[1]==0):
            p_nd = bp[nd[0],nd[1]]
            if d[nd[0],nd[1]] == d[p_nd[0],p_nd[1]]:
                self.align2trmap[p_nd[0]]=p_nd[1]
            nd = p_nd.copy()                

    # Associate the words with patterns. Checks where the patterns appeared and
    # gets the words spoken in those regions
    def __lnwordpatt__(self):
        self.lnkdata = {}
        self.lnkdata_pos={}
        # Take one pattern-id at a time
        for i in np.unique(self.patterns[:,0]):
            # select the ith patterns
            ith = self.patterns[self.patterns[:,0]==i,:]            
            temp1 = []
            temp1_pos=[]
            # Take one timeline-instance at a time
            for idx,j in enumerate(ith):
                # get indices of words occuring within the range of the patterns.
                # j+-0.5 is done to increase the pattern width by 1 sec
                selected = np.where(np.logical_and(self.walign['stime']>=max(0.,j[1]-\
                            0.25),self.walign['etime']<=min(np.max(\
                            self.walign['etime']),j[2]+0.25)))[0]
                if np.size(selected)==0:
                    continue
                # Use this selected region to extract all the words from the original
                # transcript within this region, including the words
                # that got skipped while forced alignments
                orwordidx = [self.align2trmap[item] for item in selected.tolist() if\
                    not self.align2trmap[item]==-1]
                if not orwordidx:
                    continue
                # getting the word indices from the original transcript 
                # within the pattern range
                temp = [self.trans[item].strip().lower() for \
                    item in range(min(orwordidx),max(orwordidx))+[max(orwordidx)]]
                # Get POS
                temp_pos = [ self.pos[item].strip().lower() for \
                    item in range(min(orwordidx),max(orwordidx))+[max(orwordidx)]]
                if not temp:
                    continue
                temp1.append(temp)
                temp1_pos.append(temp_pos)

            if temp1:
                if i in self.lnkdata.keys():
                    self.lnkdata[i].append(temp1)
                    self.lnkdata_pos[i].append(temp1_pos)
                else:
                    self.lnkdata[i] = temp1
                    self.lnkdata_pos[i] = temp1_pos

    # This function creates a new global variable named "selected".
    def __selectwalign__(self):
        self.selected={}
        # Take one pattern-id at a time
        for i in np.unique(self.patterns[:,0]):
            # select the ith patterns
            ith = self.patterns[self.patterns[:,0]==i,:]
            # Take one timeline-instance at a time. Idx is the index of time-instance
            for idx,j in enumerate(ith):
                # get indices of words occuring within the range of the patterns.
                # j+-0.5 is done to increase the pattern width by 1 sec
                temp = np.where(np.logical_and(self.walign['stime']>=max(0.,j[1]-\
                            0.25),self.walign['etime']<=min(np.max(\
                            self.walign['etime']),j[2]+0.25)))[0].tolist()
                if temp:
                    self.selected[(i,idx)] = temp                

'''
AMP_MT is a child of AutoMannerPlus. It assumes that the ground truth
is being read from mechanical turk annotation file.
Set raw_gt to true if you want to get all the turker annotations
'''
class AutoMannerPlus_mturk(AutoMannerPlus):
    def __readGT__(self,
        filename,
        fieldname='This body movement pattern conveys a meaning.',
        raw_gt=False
        ):
        # Most part of mechanical turk annotation is just same as participant
        # annotation, except the same video is annotated by multiple (3) turkers
        alldata = []
        data_row = ddict(list)
        data_row_mean = ddict(list)
        with open(filename,'r') as f:
            # Turker file doesn't have newline character
            gtfiledat = f.read().split('\r')
        # reading the header and calculating column id (pid) for
        # meaningfulness rating
        header = gtfiledat[0].split(',')
        pid = [idx for idx,item in enumerate(header) if item==fieldname]
        # Reading actual data
        for arow in gtfiledat[1:]:
            # Skip the random part of the data
            if 'random' in arow.lower():
                continue
            rowdat = [item if i<3 else  float(item) if item \
                else 0. for i,item in enumerate(arow.strip().split(','))]
            alldata.append(rowdat)
        # Accumulating MTurk answers
        for arow in alldata:
            data_row[arow[2]].append(arow[3:])
        # Averaging the mechanical turk answers
        for item in data_row.keys():
            data_row_mean[item] = map(int,\
                np.round(np.mean(data_row[item],axis=0)).tolist())
        # Make ground truth dict
        self.gt = {item:[data_row_mean[item][idx-3] for idx in pid]\
         for item in data_row_mean.keys()}
        # return original if necessary
        self.__gt_full__ = {akey:np.array(data_row[akey])[:,np.array(pid)-3] \
            for akey in data_row.keys()}
        if raw_gt:
            return self.__gt_full__