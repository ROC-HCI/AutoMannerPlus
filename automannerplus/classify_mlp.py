from classify import Classify

import numpy as np
import time
import sklearn as sk

# Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, ActivityRegularization 
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1,l2

# Visual
import matplotlib.pyplot as plt

'''
This is a child of classify which uses Feed-Forward NN (MLP) for regression
and classification tasks.
Apply the same hand engineered features into a multi layer perceptron
'''
class Classify_MLP(Classify):
        # Keras related
    def __init__(self,pklfilename):
        super(Classify_MLP,self).__init__(pklfilename)

    def __create_model__(self,type = 'mlp'):
        m,n = np.shape(self.x)
        
        if type == 'mlp':
            # Building a fully connected feedforward neural network
            self.model = Sequential()
            # First layer
            self.model.add(Dense(output_dim=16,input_dim=n))
            self.model.add(Activation('relu'))
            self.model.add(ActivityRegularization(l1=0.03))
            # Second layer
            self.model.add(Dense(output_dim=16))
            self.model.add(Activation('relu'))
            self.model.add(ActivityRegularization(l1=0.03))
            # output
            self.model.add(Dense(1, W_regularizer = l1(0.05)))
            self.model.add(Activation('relu'))
            # Print a summary of the neural network
            self.model.summary()
            # Compile the neural network
            self.model.compile(loss='mse',optimizer='rmsprop')
        elif type == 'mlp_classify':
            # Building a fully connected feedforward neural network
            self.model = Sequential()
            # First layer
            self.model.add(Dense(output_dim=16,input_dim=n))
            self.model.add(Activation('relu'))
            self.model.add(ActivityRegularization(l1=0.05))
            # Second layer
            self.model.add(Dense(output_dim=16))
            self.model.add(Activation('relu'))
            self.model.add(ActivityRegularization(l1=0.1))
            # output
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))
            # Print a summary of the neural network
            self.model.summary()
            # Compile the neural network
            self.model.compile(loss='mse',optimizer='rmsprop')

        elif type == 'LASSO':
            # Building a fully connected feedforward neural network
            self.model = Sequential()
            # First layer
            self.model.add(Dense(output_dim=1,input_dim=n,W_regularizer=l1(0.02)))
            self.model.add(Activation('sigmoid'))
            # Print a summary of the neural network
            self.model.summary()
            # Compile the neural network
            self.model.compile(loss='mse',optimizer='rmsprop')

    def reset_weights(self):
        weights = self.model.get_weights()

        m = len(weights)
        # replace the weights with gaussian noise
        for idx in range(m):
            mean_ = np.mean(weights[idx])
            std_ = np.std(weights[idx])
            weights[idx] = std_*np.random.randn(*np.shape(weights[idx])) + mean_
        self.model.set_weights(weights)

    # Test avg. correlation for multiple regressions
    def test_avg_corr(self,
        show_all=False,
        show_plots=False,
        task='regression', # Task can be regression or classification
        method='mlp',
        tot_iter = 5,  # Total number of repeated experiment
        paramtuning=True,
        ):

        if task == 'classification':
            if method == 'mlp':
                self.__create_model__('mlp_classify')
            elif method == 'LASSO':
                self.__create_model__('LASSO')
            # Labels for classification
            Y_ = (np.array(self.y)>3.0).astype(float)
            # Half of the data is reserved as Evaluation set
            x_train,x_test,y_train,y_test = sk.cross_validation.train_test_split(\
                self.x,Y_,test_size=0.5,random_state=int(time.time()*1000)%4294967295)

            # Training the model with 100 epochs
            auc_list=[]
            fpr = []
            tpr = []
            for i in range(tot_iter):
                # Training the model with 100 epochs
                self.reset_weights()
                self.model.fit(x_train,y_train,nb_epoch=1500,batch_size=50)
                # Get the output predictions and calculate correlation-coeff
                y_score = self.model.predict_proba(x_test)[:,0]
                # ROC Curve
                auc_list.append(sk.metrics.roc_auc_score(y_test,y_score))
                fpr_temp,tpr_temp,_ = sk.metrics.roc_curve(y_test,y_score)
                tpr.append(np.interp(np.linspace(0,1,100),fpr_temp,tpr_temp))

            print auc_list
            print 'Average AUC',np.mean(auc_list)

            plt.figure()
            plt.plot(np.linspace(0,1,100),np.mean(tpr,axis=0),label='ROC Curve')
            #plt.plot(fpr_temp,tpr_temp,label='ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.savefig('Outputs/ROC_Curve_MLP_'+self.filename[-12:]+'_'+\
                '_'+task+'.pdf',format='pdf')

        elif task == 'regression':
            self.__create_model__('mlp')
            # 1/2 of the data is reserved as Evaluation set
            x_train,x_test,y_train,y_test = \
                sk.cross_validation.train_test_split(\
                self.x,self.y,test_size=0.5,random_state=\
                int(time.time()*1000)%4294967295)

            # Calculate average correlation
            corr_list = []
            for i in range(tot_iter):
                # Training the model with 100 epochs
                self.reset_weights()
                self.model.fit(x_train,y_train,nb_epoch=100,batch_size=50)                
                y_pred = self.model.predict_proba(x_test)[:,0].tolist()
                corr_list.append(np.corrcoef(y_test,y_pred)[0,1])
            print corr_list
            print 'average correlation coefficient:',np.mean(corr_list)

        plt.show()