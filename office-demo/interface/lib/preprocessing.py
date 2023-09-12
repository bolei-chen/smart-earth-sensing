import re 
import scipy 
import pandas as pd 
import numpy as np 
import librosa 
import scipy 
from instruction import *


''' 
this is a wrapper of all the functions in this file
it returns a dictionary of 道号 to labeled pandas dataframe 
Input:  
    cable_id_to_measures: a dictionary of 道号 to measurements of sound smples where each row contains measurements of 10 seconds
    instruction_path: the path to the txt file which specifies which piece of sound sample is abnormal 
    start_time: the time when all the measurements starts 
    sample_length: how many seconds taken for each sample in the dataframe 
    measurements_in_1sec: how many mesasurements taken in one second 
''' 
def load(cable_id_to_measures, instruction_path, start_time, sample_length, measurements_in_1sec):
    print("converting measures into data frames....") 
    cable_id_to_df = get_dataframe_dict(cable_id_to_measures, sample_length, measurements_in_1sec) 
    print("done") 
    print("labeling dataframes...") 
    cable_id_to_labeled_df = label_dataframes(cable_id_to_df, instruction_path, start_time, sample_length) 
    print("done") 
    print("droping unnecessary columns...") 
    for cable_id, df in cable_id_to_labeled_df.items(): 
        cable_id_to_labeled_df[cable_id] = df.drop(['cumulative time/s'], axis=1) 
    print("done") 
    print("normalizing data frames...") 
    cable_id_to_nomalized_df = normalize(measurements_in_1sec, sample_length, cable_id_to_labeled_df) 
    print("done") 
    return cable_id_to_labeled_df 

''' 
perform normalization of df
Input:
    measurements_in_1sec: number of measurements taken in 1 second 
    sample_length: how many seconds does the sample took 
    cable_id_to_labeled_df: a dictionary of cable ids to labeled dataframes 
''' 
def normalize(measurements_in_1sec, sample_length, cable_id_to_labeled_df):
    columns_to_be_normalized = ["measure " + str(i) for i in range(1, sample_length * measurements_in_1sec + 1)] 
    col_maxs = {}
    col_mins = {}
    for cable_id, df in cable_id_to_labeled_df.items():
        maxs = {} 
        mins = {}
        for col in df.columns:
            maxs[col] = df[col].max() 
            mins[col] = df[col].min() 
        col_maxs[cable_id] = maxs 
        col_mins[cable_id] = mins 
    print("maxs and mins computed") 
    for cable_id, df in cable_id_to_labeled_df.items():
        for col in columns_to_be_normalized:
            cable_id_to_labeled_df[cable_id][col] = (df[col] - col_mins[cable_id][col]) / (col_maxs[cable_id][col] - col_mins[cable_id][col]) 
        print(cable_id, "done") 
    return cable_id_to_labeled_df 
    

''' 
returns a dictionary of cable id to its respective dataframe 
Input: 
    cable_id_to_measures: a dictionary of cable id to measures 
    sample_length: how many seconds taken for a sample in the dataframe 
''' 
def get_dataframe_dict(cable_id_to_measures, sample_length, measurements_in_1sec):
    for cable_id, measures in cable_id_to_measures.items():
        cable_id_to_measures[cable_id] = measures_to_dataframe(measures, sample_length, measurements_in_1sec) 
    return cable_id_to_measures 

''' 
Input: 
    cnvert a list of samples to a pandas dataframe 
    measures: a 1d array of samples 
    sample_time: how many seconds for each sample in the dataframe 
''' 
def measures_to_dataframe(measures, sample_time, measurements_in_1sec):
    measurements = sample_time * measurements_in_1sec
    measures = list(measures.flatten())
    chunks_of_measures = list(divide_chunks(measures, measurements))
    df = pd.DataFrame(chunks_of_measures, columns=list(map(lambda i : 'measure ' + str(i), range(1, measurements + 1)))).dropna()
    df['cumulative time/s'] = list(map(lambda i : i * sample_time, df.index))
    df['class'] = list(map(lambda i : 0, df.index)) 
    return df 
 
''' 
Input: 
    cable_id_to_dfs: a dictionary of cable id to dataframes 
    path: path for the txt file which specifies the labels
    start_time: a string of the time in the form of HH:MM:SS when all the measurements first start 
''' 
def label_dataframes(cable_id_to_dfs, path, start_time, sample_length):
    start_time = time_to_sec(start_time) 
    with open(path, 'r') as f:
        instructions = f.readlines()
    instructions = list(filter(lambda ins : ins != '', map(lambda ins : re.sub('\n| ', '', ins), instructions)))
    instructions = list(map(lambda ins : ins.split(','), instructions)) 
    instructions = list(map(lambda ins : Instruction(ins, start_time), instructions)) 
    for inst in instructions:
        df = cable_id_to_dfs[inst.cable_id]
        wanted_cumulative_times = [inst.starting_cumulative_time + i * sample_length for i in range(0, inst.duration // sample_length)] 
        for wanted_cumulative_time in wanted_cumulative_times:
            for index, row in df.iterrows():
                if row['cumulative time/s'] == wanted_cumulative_time:
                    df.at[index, 'class'] = inst.label 
        cable_id_to_dfs[inst.cable_id] = df
    return cable_id_to_dfs 
         
''' 
helper function for to_dataframe 
Input: 
    l: a 1d array of samples 
    n: size of a chunk
''' 
def divide_chunks(l, n): 
    for i in range(0, len(l), n):
        yield l[i:i + n]
          
def raw2mfcc(raw):
    b, a = scipy.signal.butter(8, 0.6)
    filtered_raw = scipy.signal.filtfilt(b, a, [m[0] for m in raw]) 
    downsampled_raw = librosa.resample(np.array(filtered_raw), orig_sr=1000, target_sr=800) 
    mfcc = np.swapaxes(librosa.feature.mfcc(y=downsampled_raw, sr=800, n_mfcc=64, hop_length=16), 0, 1) 
    return mfcc 