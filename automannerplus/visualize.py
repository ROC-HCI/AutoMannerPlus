"""
Created on Tue Jun 28 3:19:10 2016
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
"""

"""
A class for visualizing the features with respect to ground truth.
It doesn't visualize the classification results.
"""
class Visualize(object):
    def __init__(self, pklfilename = 'features_gt.pkl'):
        self.data = cp.load(open(pklfilename,'rb'))
        print 'visualizer loaded'
    def printfeaturename(self):
        for idx,feat in enumerate(self.data['featurename']):
            print idx,'\t',feat
        print '36 - end','\t','Body movement features'
    def printvideonames(self):
        for i, vidname in enumerate(np.sort(self.data['X'].keys())):
            print i,':',vidname

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
            unq_gt = np.unique(gt)
            colors = cm.cool(np.linspace(0, 1, len(unq_gt)))
            if interactive:
                plt.ion()
            for idx, item in enumerate(unq_gt):
                plt.scatter(np.array(x)[gt==item],np.array(y)[gt==item],\
                    color=colors[idx], label=item)
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
        if interactive:
            plt.ion()
        unq_y = np.unique(y)
        colors = cm.cool(np.linspace(0, 1, len(unq_y)))
        for idx, item in enumerate(unq_y):
            plt.scatter(x_project[y==item,0],x_project[y==item,1],\
                color=colors[idx], label=item)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.show()

    # Draw all the features in a 2D LDA space
    def drawlda(self,interactive=True):
        x = np.array([item for vidid_ in self.data['X'].keys() \
            for item in self.data['X'][vidid_] ])
        y = np.array([item for vidid_ in self.data['X'].keys() \
            for item in self.data['Y'][vidid_] ])
        self.lda = LDA()
        x_project=self.lda.fit_transform(x,y.tolist())

        # Now plot the features
        if interactive:
            plt.ion()
        unq_y = np.unique(y)
        colors = cm.cool(np.linspace(0, 1, len(unq_y)))
        for idx, item in enumerate(unq_y):
            plt.scatter(x_project[y==item,0],x_project[y==item,1],\
                color=colors[idx], label=item)
        plt.xlabel('First LDA Component')
        plt.ylabel('Second LDA Component')
        plt.legend()
        plt.show()