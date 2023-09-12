import sys
sys.path.append('/Users/lei/home/studyhall/smart-earth-sensing/lib')

import re 
import numpy as np 
from tqdm import tqdm 

from utils import *
from h5reader import * 
 
def h5s2raw(h5_dir_path, locus_id, maps, spec_path):
    with open(spec_path, 'r') as f:
        specs = [Spec(re.sub("\n| ", "", line).split(','), maps) for line in f.readlines()]
    feature_set, label_set = [], []
    for spec in tqdm(specs):
        for start_time in spec.duration:
            for h5_path in os.listdir(h5_dir_path):
                if start_time == h5_path[44:48]:
                    raw, _, _ =read_data(h5_dir_path + '/' + h5_path, [locus_id], [locus_id + 1]) 
                    feature_set.append(raw) 
                    label_set.append(spec.label) 
    return np.array(feature_set), np.array(label_set)
 