import sys
sys.path.append('/Users/lei/home/studyhall/smart-earth-sensing/lib') 
 
from datetime import datetime 
from glob import glob 

import torch 

from interface import *
from connection import * 
from models import CNN 
 

def main(h5_dirpath, state_dict_path, host, user, password, database, sr):
    cnn = CNN(device=torch.device("cpu"), input_shape=(1, 26, 64)) 
    cnn.load_state_dict(torch.load(state_dict_path))
    now = datetime.now() 
    sql = SQL_Connection(host, user, password, database) 
    visited = []
    while True:
        curr_h5paths = glob(h5_dirpath + '/*.h5')
        for curr_h5path in curr_h5paths:
            if curr_h5path not in visited:
                visited.append(curr_h5path) 
                pred = predict(curr_h5path, 20, cnn, sr) 
                sql.store_file((pred, now.strftime("%Y-%m-%d %H:%M:%S")))
        time.sleep(100) 
    sql.close() 
    return 0 