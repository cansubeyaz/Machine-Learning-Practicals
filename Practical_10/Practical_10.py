
import argparse
import numpy as np

from libs.features import extract_data_mlp
from libs.models import create_mpl
from libs.metrics import mlp_metrics

def parse_args():
  parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
  # parser.add_argument( '--exr', action='store', required=True )
  parser.add_argument( '--dataset', action='store', default=r'C:\Users\Lenovo\Desktop\PAML_2024\data\texture_snippets' )
  parser.add_argument( '--classes', '-c', nargs="+", default=[] )
  parser.add_argument( '--split', action='store', type=float, default=0.3 )
  parser.add_argument( '--orient', action='store', type=int, default=8 )
  parser.add_argument( '--ppc', action='store', type=int, default=8 )
  parser.add_argument( '--cpb', action='store', type=int, default=1 )
  parser.add_argument( '--nclusters', action='store', type=int, default=32 )
  # parser.add_argument( '--C', action='store', type=float, default=1.0 )
  parser.add_argument( '--radius', action='store', type=int, default=1 )
  parser.add_argument( '--npoints', action='store', type=int, default=8 )
  parser.add_argument( '--nbins', action='store', type=int, default=64 )
  parser.add_argument( '--method', action='store', default='default' )
  parser.add_argument( '--hidden', action='store', type=int, nargs="+", default=[64] )
  parser.add_argument( '--epochs', action='store', type=int, default=10 )
  return parser.parse_args()

if __name__ == '__main__':
    # get the flags.
    flags = parse_args()
    # get the features
    Xt, Lt, Xe, Le = extract_data_mlp( flags )
    # get the mlp model
    model = create_mpl( flags, len( np.unique( Lt ) ) )
    model.fit( Xt, Lt )
    # evaluate the model
    cls = model.predict( Xe )
    probs = model.predict_proba( Xe )
    mlp_metrics( Le, cls, probs[:,1] )

    ## python .\Practical_10.py --classes spots plain plaid