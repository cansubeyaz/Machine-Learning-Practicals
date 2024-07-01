"""
  Logistic Regression
  Saving models and loading models.
  Segmenting an image
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

from skimage.io import imread, imsave
from skimage.color import rgb2lab

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve as prc

from libs.regression import LogisticRegression
from libs.extraction import extract_train_eval_files

def parse_args():
  parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
  parser.add_argument( '--exr', action='store', required=True )
  parser.add_argument( '--dataset', action='store', default=r'C:\Users\Lenovo\Desktop\PAML_2024\data\colour_snippets' )
  parser.add_argument( '--classes', '-c', nargs="+", default=[] )
  parser.add_argument( '--split', action='store', type=float, default=0.3 )
  parser.add_argument( '--imax', action='store', type=int, default=10 )
  parser.add_argument( '--logreg', action='store', default='logreg_model.pkl' )
  parser.add_argument( '--image', action='store', default=r'C:\Users\Lenovo\Desktop\PAML_2024\Practical_09\sweetpepper.png' )
  return parser.parse_args()

# run with
# python solutions/Esra/Practical_09.py --exr exr0 --classes green red


def exr0( flags ):
  t_files, t_labs, e_files, e_labs = extract_train_eval_files( flags.dataset, flags.split, flags.classes, extension='png' )
  def read_files_to_matrix( filenames, labels ):
    first = True
    for f, lb in zip( filenames, labels ):
      i = imread( f )/255.
      l = rgb2lab( i )
      if first:
        rgb = i.reshape( -1, 3 )
        lab = l.reshape( -1, 3 )
        labels = [lb]*rgb.shape[0]
        first = False
      else:
        r = i.reshape( -1, 3 )
        rgb = np.vstack( (rgb, i.reshape( -1, 3 )) )
        lab = np.vstack( (lab, l.reshape( -1, 3 )) )
        labels += [lb]*r.shape[0]
    return rgb, lab, np.array( labels )
  t_rgb_pixels, t_lab_pixels, t_labels = read_files_to_matrix( t_files, t_labs )
  e_rgb_pixels, e_lab_pixels, e_labels = read_files_to_matrix( e_files, e_labs )
  # print( t_rgb_pixels.shape, t_lab_pixels.shape, t_labels.shape )
  # print( e_rgb_pixels.shape, e_lab_pixels.shape, e_labels.shape )
  # get just the a channel for red and green
  # t_lab_pixels = t_lab_pixels[:,1].reshape( -1, 1 )
  # e_lab_pixels = e_lab_pixels[:,1].reshape( -1, 1 )
  ##########################
  # don't do this first!
  mu = t_lab_pixels.mean( axis=0 )
  st = t_lab_pixels.std( axis=0 )
  t_lab_pixels = (t_lab_pixels-mu)/st
  e_lab_pixels = (e_lab_pixels-mu)/st
  ###################
  # now logistic regression
  obj = LogisticRegression( imax=flags.imax )
  obj.fit( t_lab_pixels, t_labels )
  probs, cls = obj.predict( e_lab_pixels )
  def logreg_metrics( labels, preds, probs ):
    # now score the stuff
    acc = accuracy_score( labels, preds )
    print( f'Accuracy of Logistic Regression is: {acc:0.04f}' )
    cnf = confusion_matrix( labels, preds, normalize='true' )
    print( cnf )
    cnfacc = cnf.diagonal().sum()/2
    print( f'confusiong accuracy {cnfacc}' )
    p, r, t = prc( labels, probs )
    f1 = 2*p*r/(p+r+0.0000001)
    # plot the precision recall and the point where F1-score is at it's maximum.
    am = np.argmax( f1 )
    plt.figure()
    plt.plot()
    plt.plot( r, p )
    plt.plot( r[am], p[am], 'r*' )
    plt.title( f'PR Curve: F1-score of {f1[am]}' )
    plt.show()
    return cnfacc, f1[am], t[am]
  labcnf, labf1, thresh = logreg_metrics( e_labels, cls, probs )

  # RGB
  ##########################
  # don't do this first!
  mu_rgb = t_rgb_pixels.mean( axis=0 )
  st_rgb = t_rgb_pixels.std( axis=0 )
  t_rgb_pixels = (t_rgb_pixels-mu_rgb)/st_rgb
  e_rgb_pixels = (e_rgb_pixels-mu_rgb)/st_rgb
  ###################
  obj1 = LogisticRegression( imax=flags.imax )
  obj1.fit( t_rgb_pixels, t_labels )
  probs, cls = obj1.predict( e_rgb_pixels )
  rgbcnf, rgbf1, rgb_thresh = logreg_metrics( e_labels, cls, probs )

  # get the best model and save it
  dit = {}
  dit['logreg'] = obj
  dit['mu'] = mu
  dit['std'] = st
  dit['thresh'] = thresh
  with open( flags.logreg, 'wb' ) as fid:
    pickle.dump( dit, fid )

def exr1( flags ):
  with open( flags.logreg, 'rb' ) as fid:
    info = pickle.load( fid )
  model = info['logreg']
  mu = info['mu']
  st = info['std']
  thresh = info['thresh']
  img = imread( flags.image )
  # plt.figure()
  # plt.imshow( img )
  # plt.show()
  # now segment the image
  r, c = img.shape[0], img.shape[1]
  # reshape the image
  pxl = img.reshape( -1, 3 )
  # predict it
  probs, preds = model.predict( pxl )
  # using the 0.5 classification
  img = np.array( preds ).reshape( (r,c) )*255
  print( img.shape )
  imsave( 'classified.png', img.astype( np.uint8 ) )
  # using the threshold
  probs = np.array( probs ).reshape( (r,c) )
  img = np.zeros( (r,c) )
  img[probs<=thresh] = 255
  imsave( 'f1score.png', img.astype( np.uint8 ) )

if __name__ == "__main__":
  flags = parse_args()
  #
  if flags.exr == 'exr0': ## python .\Practical_09.py --exr exr0 --classes green red
    exr0( flags )
  if flags.exr == 'exr1': ## python .\Practical_09.py --exr exr1 --classes green red
    exr1( flags )
