# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:52:24 2016
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
"""
import time
import pandas as pd

class Word2Vec(object):
    # Initialize
    def __init__(self):
        self.whash = {}
        self.ihash = {}
        self.vlist = []
    
    # Read the w2v contents from text or pickled file
    # returns a Vector(dim=300) list of words, a hash
    # of words giving indices, and a hash of indices
    # giving words
    def load(self,filename='order4.txt',silent=False):
        t = time.time()
        # Loading from text is time consuming
        if not silent:print 'Loading from file ... please wait'
        x = pd.read_csv(filename,delim_whitespace=True)
        if not silent:print 'Converting to correct structure'
        temp = x.as_matrix()
        self.whash = {item:idx for idx,item in enumerate(temp[:,0])}
        self.ihash = {idx:item for idx,item in enumerate(temp[:,0])}
        self.vlist = temp[:,1:].astype('float64')
        elapsed = time.time() - t
        if not silent:print 'Data loaded in', elapsed, 'sec'
    
    # get vector from word
    def v(self,word):
        if not word in self.whash:
            return None
        else:
            return self.vlist[self.whash[word],:]
    
    # Calculate similarity of a vecotor with the list of vectors
    def sim_score(self,vec):
        return self.vlist.dot(vec[None].T).flatten()
    
    # Get the top n indices similar to a vector
    def topsim(self,vec,n=10):
        score = self.sim_score(vec)
        return score.argsort()[-n:][::-1]
        
    # Get the top n words similar to a vector
    def w(self,vec,n=10):
        idx = self.topsim(vec,n)
        list_w = []
        for i in idx:
            list_w.append(self.ihash[i])
        return list_w
        