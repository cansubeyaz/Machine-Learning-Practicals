
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve as prc

class pr_curve():
    def __init__(self, epsilon=0.00000001 ):
        self.eps = epsilon

    def calculate_statistic(self, labels, predictions ):
        TP = np.sum( np.logical_and( labels, predictions ) )
        FP = np.sum( np.logical_and( np.logical_not( labels ), predictions ) )
        MD = np.sum( np.logical_and( labels, np.logical_not( predictions ) ) )
        P = TP/(TP+FP+self.eps)
        R = TP/(TP+MD+self.eps)
        F1 = 2*P*R/(P+R+self.eps)
        return P, R, F1

    def __call__(self, labels, scores ):
        ssorted = np.unique( np.sort( scores.reshape( -1 ) ) )
        P = np.zeros( (ssorted.shape[0],) )
        R = np.zeros( (ssorted.shape[0],) )
        F1 = np.zeros( (ssorted.shape[0],) )
        for i, s in enumerate( ssorted ):
            P[i], R[i], F1[i] = self.calculate_statistic( labels, scores>=s )

        i = np.argmax( F1 )
        plt.figure()
        plt.plot(R, P)
        plt.plot(R[i], P[i], 'r.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall plot - F1-score {:0.03f}'.format(F1[i]))
        plt.show()
        # print the best value
        print(
            '| Precision: {:0.07f}\n'.format(P[i]),
            '| Recall: {:0.07f}\n'.format(R[i]),
            '| F1 Score: {:0.07f}\n'.format(F1[i]),
            '| Threshold: {:0.07f}\n'.format(ssorted[i]) )
        return P[i], R[i], F1[i], ssorted[i]


def svm_metrics( eval_data, model, labels, name ):
    # predict the evaluation stuff
    preds = model.predict( eval_data )
    # get the accuracy score
    acc = accuracy_score( np.array( labels ).reshape( -1,1 ), preds.reshape( -1, 1) )
    print( f'The accuracy score of the {name} is {acc:0.4f}' )
    print( confusion_matrix( labels, preds, normalize='true' ) )
    if np.unique( labels ).shape[0] == 2:
        scores = model.decision_function( eval_data )
        p,r,t = prc( labels, scores )
        f1 = 2*p*r/(p+r+0.0000000001)
        am = np.argmax( f1 )
        plt.figure()
        plt.plot( r, p )
        plt.plot( r[am], p[am], 'r*' )
        plt.title( f'PR curve {name}: F1 of {f1[am]}' )
        plt.show()

def mlp_metrics( labels, cls, probs ):
    acc = accuracy_score( labels, cls )
    print( f'The accuracy of this mlp is {acc:0.3f}' )
    cnf = confusion_matrix( labels, cls, normalize='true' )
    cnfacc = cnf.diagonal().sum()/float( cnf.shape[0] )
    print( f'Confusion matrix {cnfacc:0.3f}' )
    print( cnf )
    if len( np.unique( labels ) ) == 2:
        p, r, t = prc( labels, probs )
        f1 = 2*p*r/(p+r+0.000000001)
        am = np.argmax( f1 )
        plt.figure()
        plt.plot( r, p )
        plt.plot( r[am], p[am], 'r*' )
        plt.title( f'PR Curve: best f1={f1[am]:0.3f}')
        plt.tight_layout()
        plt.show()