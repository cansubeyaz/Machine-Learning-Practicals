
import numpy as np
import matplotlib.pyplot as plt

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