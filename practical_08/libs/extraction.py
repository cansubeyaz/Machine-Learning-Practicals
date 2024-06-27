
import glob
import os

from sklearn.model_selection import train_test_split

def extract_train_eval_files( loc, split, classes, extension='jpg' ):
    tfiles, efiles, tlabs, elabs = [], [], [], []
    eval_label = 0
    for tex in sorted( os.listdir( loc ) ):
        if tex not in classes:
            continue
        files = sorted( glob.glob( os.path.join( loc, tex, '*.'+extension ) ) )
        tr, ev = train_test_split( files, test_size=split, shuffle=False )
        tfiles += tr
        efiles += ev
        tlabs += [eval_label]*len( tr )
        elabs += [eval_label]*len( ev )
        eval_label += 1
    return tfiles, tlabs, efiles, elabs