# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:44:10 2016
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
"""
import numpy as np
import nltk
from Word2Vec import Word2Vec
import csv
import cPickle as cp
from collections import defaultdict as ddict
from scipy.signal import resample
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class AutoMannerPlus(object):
    ''' 
    A class extracting features for classifying gestures as mannerism or meaningful.
    Class global variables:
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
                         value is set to -1.
    gt                 : A dictionary from vidid (string) to the selected ground truth data.
    patterns           : A numpy array of pattern# and its start time and end time (in sec)
    vidid              : id (string) of the current video
    lnkdata            : List containing a patternid and the words spoken within that pattern. The
                         list is grouped in such a way that lnkdata[0] with return the first set of
                         patterns (not guaranteed that the first is 0).
    lnkdata_pos        : Similar to lnkdata but contains POS
    w2vdata            : Similar to lnkdata but contains w2v info
    alignpath, 
    timepath, 
    trasnpath          : various pathnames
    selected           : A dictionary from (pat,inst) tuple to indices of walign that falls within that
                         inst's time period.
'''
    # Provide the filenames: pattern-timeline file, aligned transcript file
    def __init__(self,
        gtfilename, # ground truth filename
        alignpath,  # path where alignfile resides
        timepath,   # path where timeline file resides (provide the folder containing the video\
                    # name folder)
        prosodypath # path where the prosody files are located
        ):
        self.__readGT__(gtfilename)
        self.alignpath = alignpath
        self.timepath = timepath
        self.prosodypath = prosodypath
        self.filler = ['uhh', 'um', 'uh', 'umm', 'like','say','basically',\
        'so', 'ah', 'umhum', 'uhum', 'uhm','oh','oho','oooh','huh','ah','aha']
    
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

    # A convenience function to view the ground truth data
    def viewgt(self):
        patlist = np.unique(self.patterns[:,0])
        print 'patternID','\t','GT_Score'
        for idx,apat in enumerate(patlist):
            print apat, '\t','\t',self.gt[self.vidid][idx]

    # Read files in full detail. w2v is a preloaded Word2Vector object 
    def readEverything(self,transpath,vidid,w2v):
        self.transpath = transpath
        if isinstance(w2v,Word2Vec) and len(w2v.vlist)>480000:
            self.w2v = w2v
        else:
            raise ValueError('w2v must be a loaded Word2Vec object')
        self.vidid = vidid
        # Read the files
        self.__read_align__(self.alignpath+vidid+'.txt')
        self.__read_time__(self.timepath+vidid+'/timeline_'+vidid+'.csv')
        self.__read_trans__(self.transpath+vidid+'.txt')
        self.__read_prosody__()
        # Build the links with transcript data. This part is time consuming
        self.__buildalign2trmap__()
        self.__lnwordpatt__()
        self.__selectwalign__()
        
    # Faster read without the transcription data. Some features don't need transcripts.
    # However, transcript data requires a time-consuming alignment process which could be avoided
    # in a faster group of feature extraction.
    # Please note, this function doesn't prepare the variables align2trmap, lnkdata, lnkdata_pos and
    # w2vdata. So, after calling this function, those variables are either unavailable or non-updated
    # However, this creates a new variable: selected
    def readfast(self,vidid):
        self.vidid = vidid
        # Read the files
        self.__read_align__(self.alignpath+vidid+'.txt')
        self.__read_time__(self.timepath+vidid+'/timeline_'+vidid+'.csv')
        self.__read_prosody__()
        self.__selectwalign__()
        
        
    # Extract features from the aligned transcript data
    # Features extracted in this function are as follows (per instance):
    # mean_wrdtime, mean_filltime, mean_pausetime, #w, #fill, #pause,
    # word%_inst, filler%_inst, pause%_inst & the prosody features
    def extractfeaturesfast(self):
        def getgt(vidid,i):
            return self.gt[vidid][np.where(np.unique(self.patterns[:,0])==i)[0][0]]
        def id(pat,inst):
            return np.where(self.patterns[:,0]==pat)[0][inst]
        featurelist = {}
        gtlist={}
        # for every pattern i and instance j
        for i,j in self.selected.keys():
            if getgt(self.vidid,i)==0.:
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
            # Enlisting the features
            featurelist[(i,j)]=[\
                        # ================== Disfluency features (9) ===================
                        np.mean(wrdTime)if wrdTime else 0.0, # Average time to speak word
                        np.mean(fillTime)if fillTime else 0.0, # Average time to speak filler word
                        np.mean(spTime) if spTime else 0.0,    # Average pause length
                        len(wrdTime) if wrdTime else 0.0,      # Total number of words
                        len(fillTime) if fillTime else 0.0,    # Total number of filler words
                        len(spTime) if spTime else 0.0,        # Total number of pauses
                        np.sum(wrdTime)/instLen if wrdTime else 0.0, # % of word per instance
                        np.sum(fillTime)/instLen if fillTime else 0.0, # % of filler per instance
                        np.sum(spTime)/instLen if spTime else 0.0, # % of filler per instance
                        # ==================== Prosody Features (26) =====================
                        np.mean(loud),                          # Mean loudness
                        np.min(loud),                           # Minimum loudness
                        np.max(loud),                           # Maximum loudness
                        np.max(loud) - np.min(loud),            # loudness range
                        np.std(loud),                           # loudness standard deviation
                        np.mean(pitch),                         # Mean pitch
                        np.min(pitch),                          # Minimum pitch
                        np.max(pitch),                          # Maximum pitch
                        np.max(pitch) - np.min(pitch),          # pitch range
                        np.std(pitch),                          # pitch standard deviation
                        form_min[0],                            # Minimum first formant
                        form_min[1],                            # Minimum second formant
                        form_min[2],                            # Minimum third formant
                        form_max[0],                            # Maximum first formant
                        form_max[1],                            # Maximum second formant
                        form_max[2],                            # Maximum third formant
                        form_mean[0],                           # Mean first formant
                        form_mean[1],                           # Mean second formant
                        form_mean[2],                           # Mean third formant
                        form_std[0],                            # first formant std
                        form_std[1],                            # second formant std
                        form_std[2],                            # third formant std
                        form_range[0],                          # first formant range
                        form_range[1],                          # second formant range
                        form_range[2],                          # third formant range
                        np.count_nonzero(pitch)/len(pitch)      # percent unvoiced
                        ]
            gtlist[(i,j)]=getgt(self.vidid,i)
        return featurelist,gtlist
    def featurename(self):
        return ['Average time to speak word','Average time to speak filler word',\
'Average pause length','Total number of words','Total number of filler words',\
'Total number of pauses','% of word per instance','% of filler per instance',\
'% of filler per instance','Mean loudness','Minimum loudness','Maximum loudness',\
'loudness range','loudness standard deviation','Mean pitch','Minimum pitch',\
'Maximum pitch','pitch range','pitch standard deviation','Minimum first formant',\
'Minimum second formant','Minimum third formant','Maximum first formant',\
'Maximum second formant','Maximum third formant','Mean first formant',\
'Mean second formant','Mean third formant','first formant std','second formant std',\
'third formant std','first formant range','second formant range',\
'third formant range','percent unvoiced']
        
    def __check_fillerness__(self,wrd):
        raise NotImplementedError
    
    # This is the full version of feature extraction. call it only if you used the readEverything
    # function earlier
    def extractAllFeatures(self):
        raise NotImplementedError('Not implemented yet')
        
    # View the length of the spoken words
    def viewwordlen(self):
        # for every instances in the patterns
        for i,j in self.selected.keys():
            print i,j,'=',[(self.walign['word'][item],self.walign['etime'][item]-\
            self.walign['stime'][item]) for item in self.selected[i,j]]

    # Calculate the contextual similarity of the patterns. Calling this function
    # requires full file read, not the faster version
    def calcContextSim(self,simType='w2v'):
        patlist = np.unique(self.patterns[:,0])
        patsim=[]
        # Calculate similarity through w2v
        if simType == 'w2v':
            # for every pattern
            for idx in range(len(self.lnkdata)):
                instSim=[]
                # for every instance in the pattern
                for i in range(len(self.w2vdata[idx])):
                    for j in range(i):
                        # In order to calculate similarity of a set of n 300 dimensional vectors 
                        # another set of k 300 dimensional vectors, we apply matrix product
                        instSim.extend(self.w2vdata[idx][i].dot \
                            (self.w2vdata[idx][j].T).flatten().tolist())
                # participant's annotation
                gtvalue = self.gt[self.vidid][np.where(patlist == self.lnkdata[idx][0][0])[0][0]]
                if not instSim:
                    continue
                simdat = (self.vidid,self.lnkdata[idx][0][0],np.mean(instSim),gtvalue)
                patsim.append(simdat)
                print simdat
        # Calculate similarity through parts of speech
        elif simType == 'pos':
            # for every pattern
            for idx,apat in enumerate(self.lnkdata_pos):
                instSim=[]
                # for every instance in the pattern
                for i in range(len(apat)):
                    for j in range(i):
                        # calculate similarity by pos intersection count / pos union count
                        instSim.append(float(len(np.intersect1d(self.lnkdata_pos[idx][i][1:],\
                        self.lnkdata_pos[idx][j][1:])))/ float(len(np.union1d(\
                        self.lnkdata_pos[idx][i][1:],self.lnkdata_pos[idx][j][1:]))))
                # participant's annotation
                gtvalue = self.gt[self.vidid][np.where(patlist == self.lnkdata[idx][0][0])[0][0]]
                if not instSim:
                    continue
                simdat = (self.vidid,self.lnkdata[idx][0][0],np.mean(instSim),gtvalue)
                patsim.append(simdat)
                print simdat            
        return patsim

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
    
    # Associate the words with patterns. Checks where the patterns appeared and
    # gets the words spoken in those regions
    def __lnwordpatt__(self):
        self.lnkdata = []
        self.lnkdata_pos=[]
        self.w2vdata = []
        # Take one pattern-id at a time
        for i in np.unique(self.patterns[:,0]):
            # select the ith patterns
            ith = self.patterns[self.patterns[:,0]==i,:]            
            temp1 = []
            temp1_pos=[]
            temp_v=[]
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
                temp1.append([i]+temp)
                temp1_pos.append([i]+temp_pos)
               
                # Check Warning cause: Done. Nothing to do
                arr = np.array([self.w2v.v(item) for item in temp if item and not \
                    (self.w2v.v(item) == None)])
                if not len(np.shape(arr))==2:
                    continue
                temp_v.append(arr)
            if temp1:
                self.lnkdata.append(temp1)
                self.lnkdata_pos.append(temp1_pos)
            if temp_v:
                self.w2vdata.append(temp_v)

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
                    n = 3
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

    # Create a map frm the alignment word to the transcript word (ignore sp) using dynamic
    # programming. This is important because the aligned transcript file has missing words.
    # Also, the sentence boundary and other POS information is relative to the original
    # transcript.   
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
            nd =p_nd.copy()        
                
    # Read a timeline data: patternID, startsec,endsec
    def __read_time__(self,filename):
        with open(filename,'r') as f:
            f.readline()
            self.patterns = np.array([[rows[0]]+rows[2:].strip().split(',')[1:]\
                for rows in f],dtype=int)

class visualize(object):
    """A class for visualizing the features with respect to ground truth.
    """
    def __init__(self, pklfilename = 'features_gt.pkl'):
        self.data = cp.load(open(pklfilename,'rb'))
        print 'visualizer loaded'
    def printfeaturename(self):
        for idx,feat in enumerate(self.data['featurename']):
            print idx,'\t',feat
    def printvideonames(self):
        for vidname in np.sort(self.data['X'].keys()):
            print vidname
    # provide indices of two features and a videoid to plot wrt gt
    def draw2features(self,featlist,vidid='all',interactive=True):
        if not len(featlist)==2:
            raise ValueError("featlist must contain indices of two features")
        else:
            if not vidid=='all':
                x = [item[featlist[0]] for item in self.data['X'][vidid]]
                y = [item[featlist[1]] for item in self.data['X'][vidid]]
                gt = [item for item in self.data['Y'][vidid]]
            else:
                x = [item[featlist[0]] for vidid_ in self.data['X'].keys() \
                    for item in self.data['X'][vidid_] ]
                y = [item[featlist[1]] for vidid_ in self.data['X'].keys() \
                    for item in self.data['X'][vidid_] ]
                gt = [item for vidid_ in self.data['X'].keys() for item in \
                    self.data['Y'][vidid_] ]
            # Now plot the features
            leg = ['red','blue','green','yellow','cyan','magenta','black']
            if interactive:
                plt.ion()
            for idx, item in enumerate(np.unique(gt)):
                plt.scatter(np.array(x)[gt==item],np.array(y)[gt==item],\
                    c=leg[idx], label=idx+1)
                plt.xlabel(self.data['featurename'][featlist[0]])
                plt.ylabel(self.data['featurename'][featlist[1]])
            plt.legend()
            plt.show()

    # Draw all the features in a 2D PCA space
    def drawpca(self,interactive=True):
        x = np.array([item for vidid_ in self.data['X'].keys() \
            for item in self.data['X'][vidid_] ])
        y = np.array([item for vidid_ in self.data['X'].keys() \
            for item in self.data['Y'][vidid_] ])
        pca = PCA(n_components=2)
        x_project = pca.fit_transform(x)
        # Now plot the features
        leg = ['red','blue','green','yellow','cyan','magenta','black']
        if interactive:
            plt.ion()
        for idx, item in enumerate(np.unique(y)):
            plt.scatter(x_project[y==item,0],x_project[y==item,1],\
                c=leg[idx], label=idx+1)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.show()        

    # Draw all the features in a 2D PCA space
    def drawlda(self,interactive=True):
        x = np.array([item for vidid_ in self.data['X'].keys() \
            for item in self.data['X'][vidid_] ])
        y = np.array([item for vidid_ in self.data['X'].keys() \
            for item in self.data['Y'][vidid_] ])
        lda = LDA()
        x_project=lda.fit_transform(x,y)
        # Now plot the features
        leg = ['red','blue','green','yellow','cyan','magenta','black']
        if interactive:
            plt.ion()
        for idx, item in enumerate(np.unique(y)):
            plt.scatter(x_project[y==item,0],x_project[y==item,1],\
                c=leg[idx], label=idx+1)
        plt.xlabel('First LDA Component')
        plt.ylabel('Second LDA Component')
        plt.legend()
        plt.show()        

# In this first approach, we assumed that if the various time-instances where
# the same pattern occurred shows similiar words, then it means a good pattern
# This part generates a similarity score between various time-instances of the same
# pattern and calculate the correlation between the score and the ground truth
# However, the low value of the correlation implies this approach was not good
def firstapproach():
    alignpath = '/Users/itanveer/Data/ROCSpeak_BL/features/alignments/'
    timepath = '/Users/itanveer/Data/ROCSpeak_BL/Original_Data/Results/'
    transpath = '/Users/itanveer/Data/ROCSpeak_BL/Ground-Truth/Transcripts/'
    gtfile = '/Users/itanveer/Data/ROCSpeak_BL/Ground-Truth/participants_ratings.csv'
    prosodypath = '/Users/itanveer/Data/ROCSpeak_BL/features/prosody/'
    
    w2v = Word2Vec()
    w2v.load()
    amp = AutoMannerPlus(gtfile,alignpath,timepath,prosodypath)
    vid = ['34.1','35.2','36.1','37.2','38.1','39.2','40.1','41.2','42.1','44.1','45.2',
            '46.1','47.2','48.1','49.2','50.1','51.2','52.1','53.2','54.1','55.2','56.1','57.2',
            '58.1','59.2','60.1','61.2','62.1']
    score_w2v=[]
    score_pos=[]
    lnkdat_words=ddict(list)
    lnkdat_pos=ddict(list)
    for avid in vid:
        print 'processing ... ',avid
        # Read the necessary files
        amp.readEverything(transpath,avid,w2v)
        # Calculate similarity
        score_w2v.append(amp.calcContextSim('w2v'))
        score_pos.append(amp.calcContextSim('pos'))
        # Add linked words for saving
        lnkdat_words[avid].extend([item for item in amp.lnkdata[:]])
        lnkdat_pos[avid].extend([item for item in amp.lnkdata_pos[:]])
    x_w2v = np.array([subitem for item in score_w2v for subitem in item],dtype='a5,i2,f8,f2')
    x_pos = np.array([subitem for item in score_pos for subitem in item],dtype='a5,i2,f8,f2')
    cp.dump({'simscore_w2v':x_w2v,'simscore_pos':x_pos,'words':lnkdat_words,'pos':lnkdat_pos,\
        'w2vCorr':np.corrcoef(x_w2v['f2'],x_w2v['f3']),'posCorr':np.corrcoef(\
        x_pos['f2'],x_pos['f3'])},open('output.pkl', 'wb'))
    print 'w2v',np.corrcoef(x_w2v['f2'],x_w2v['f3'])
    print 'pos',np.corrcoef(x_pos['f2'],x_pos['f3'])    

# In this approach, we just extract a number of features and then try to cluster
# them using PCA and LDA. We can also train a classifier to predict the ground truth.
def secondapproach():
    alignpath = '/Users/itanveer/Data/ROCSpeak_BL/features/alignments/'
    timepath = '/Users/itanveer/Data/ROCSpeak_BL/Original_Data/Results/'
    gtfile = '/Users/itanveer/Data/ROCSpeak_BL/Ground-Truth/participants_ratings.csv'
    prosodypath = '/Users/itanveer/Data/ROCSpeak_BL/features/prosody/'
    
    amp = AutoMannerPlus(gtfile,alignpath,timepath,prosodypath)
    vid = ['34.1','35.2','36.1','37.2','38.1','39.2','40.1','41.2','42.1','44.1','45.2',
            '46.1','47.2','48.1','49.2','50.1','51.2','52.1','53.2','54.1','55.2','56.1','57.2',
            '58.1','59.2','60.1','61.2','62.1']
    X_data = ddict(list)
    Y_data = ddict(list)
    vidid=[]
    for avid in vid:
        print 'processing ...',avid
        amp.readfast(avid)
        features,gt_ = amp.extractfeaturesfast()
        for i,j in features.keys():
            X_data[avid].append(features[i,j])
            Y_data[avid].append(gt_[i,j])
    print 'Dump all data to features_gt.pkl file'
    cp.dump({'X':X_data,'Y':Y_data,'featurename':amp.featurename()},open('features_gt.pkl','wb'))

if __name__=='__main__':
    secondapproach()
