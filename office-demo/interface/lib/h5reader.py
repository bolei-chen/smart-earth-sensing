import h5py
from tqdm import tqdm 
import numpy as np
from obspy.core import UTCDateTime
import glob
import os

def read_meta_data(path):
    file = h5py.File(path, 'r')
    space_interval = file['Acquisition'].attrs['SpatialSamplingInterval']
    FileSampleRate = file['Acquisition/Raw[0]'].attrs['OutputDataRate']
    raw_all_datastarttime = file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime']
    raw_all_datastarttime = list(str(raw_all_datastarttime))
    for i in range(2):
        raw_all_datastarttime.pop(0)
    for m in range(7):
        raw_all_datastarttime.pop(-1)
    raw_all_datastarttime = "".join(raw_all_datastarttime)
    raw_all_datastarttime = UTCDateTime(raw_all_datastarttime, precision=2)
    # endtime=UTCDateTime(endtime)
    raw_all_dataendtime = file['Acquisition/Raw[0]/RawData'].attrs['PartEndTime']
    raw_all_dataendtime = list(str(raw_all_dataendtime))
    for i in range(2):
        raw_all_dataendtime.pop(0)
    for m in range(7):
        raw_all_dataendtime.pop(-1)
    raw_all_dataendtime = "".join(raw_all_dataendtime)
    raw_all_dataendtime = UTCDateTime(raw_all_dataendtime)
    return space_interval, FileSampleRate, raw_all_datastarttime, raw_all_dataendtime


def read_time_data(path, startlocus, endlocus):
    file = h5py.File(path, 'r')
    # cal the index of locus
    startlocus_index = startlocus
    endlocus_index = endlocus
    raw_CHOSE_data = file['Acquisition/Raw[0]/RawData'][:,
                                                        startlocus_index:endlocus_index+1]
  # choose the correct dataset
    # which storage the experiment data
    # Set the index caluted before and extract the data
    raw_CHOSE_data_array = np.array(raw_CHOSE_data)
    dimes = raw_CHOSE_data_array.shape
    return raw_CHOSE_data

def read_data(path, locus_start, locus_end):
    files = sorted(glob.glob(path))
    file_index = 0
    raw_chose_data_arr = []
    final_array = []
    FileSampleRate = 0
    space_interval = 0
    for file in tqdm(files):
        file = h5py.File(file, 'r')
        if file_index == 0:
            space_interval = file['Acquisition'].attrs['SpatialSamplingInterval']
            FileSampleRate = file['Acquisition/Raw[0]'].attrs['OutputDataRate']
            for l in range(len(locus_start)):
                if l == 0:
                    locus_start_index = int(locus_start[l]/space_interval)
                    locus_end_index = int(locus_end[l]/space_interval)
                    raw_chose_data = file['Acquisition/Raw[0]/RawData'][:,
                                                                        locus_start_index:locus_end_index]
                    raw_chose_data_arr = np.array(raw_chose_data)
                else:
                    locus_start_index = int(locus_start[l] / space_interval)
                    locus_end_index = int(locus_end[l] / space_interval)
                    raw_chose_data = file['Acquisition/Raw[0]/RawData'][:,
                                                                        locus_start_index:locus_end_index]
                    raw_chose_data_arr = np.hstack(
                        (raw_chose_data_arr, raw_chose_data))
            # raw_chose_data=np.array(raw_chose_data)
            final_array = np.array(raw_chose_data_arr)
        else:
            for l in range(len(locus_start)):
                if l == 0:
                    locus_start_index = int(locus_start[l] / space_interval)
                    locus_end_index = int(locus_end[l] / space_interval)
                    raw_chose_data = file['Acquisition/Raw[0]/RawData'][:,
                                                                        locus_start_index:locus_end_index]
                    raw_chose_data_arr = np.array(raw_chose_data)
                else:
                    locus_start_index = int(locus_start[l] / space_interval)
                    locus_end_index = int(locus_end[l] / space_interval)
                    raw_chose_data = file['Acquisition/Raw[0]/RawData'][:,
                                                                        locus_start_index:locus_end_index]
                    raw_chose_data_arr = np.hstack(
                        (raw_chose_data_arr, raw_chose_data))
            # raw_chose_data=np.array(raw_chose_data)
            final_array = np.vstack((final_array, raw_chose_data_arr))

        #del file
        file_index += 1
        #dimes = data_array.shape
    return final_array, space_interval, FileSampleRate


def TimeSegment(rawDataArray, lenWindow):
    dataMatrix = rawDataArray.transpose()

    lenTime = dataMatrix.shape[1]
    lenPad = int((lenTime + lenWindow - 1) // lenWindow * lenWindow - lenTime)

    dataMatrix = np.pad(dataMatrix, ((0, 0), (0, lenPad)),
                        'constant', constant_values=0)
    waveTensor = dataMatrix.reshape(dataMatrix.shape[0], int(
        dataMatrix.shape[1] / lenWindow), int(lenWindow))
    return waveTensor

def read_data_v2(path, locus_start, locus_end):
    files = sorted(glob.glob(path))
    file_index = 0
    raw_chose_data_arr = []
    final_array = []
    FileSampleRate = 0
    space_interval = 0
    for file_sample in files:
        with h5py.File(file_sample, 'r') as file:
            if file_index == 0:
                space_interval = file['Acquisition'].attrs['SpatialSamplingInterval']
                FileSampleRate = file['Acquisition/Raw[0]'].attrs['OutputDataRate']
                for l in range(len(locus_start)):
                    if l == 0:
                        locus_start_index = int(locus_start[l]/space_interval)
                        locus_end_index = int(locus_end[l]/space_interval)
                        raw_chose_data = file['Acquisition/Raw[0]/RawData'][:,
                                                                            locus_start_index:locus_end_index]
                        raw_chose_data_arr = np.array(raw_chose_data)
                    else:
                        locus_start_index = int(
                            locus_start[l] / space_interval)
                        locus_end_index = int(locus_end[l] / space_interval)
                        raw_chose_data = file['Acquisition/Raw[0]/RawData'][:,
                                                                            locus_start_index:locus_end_index]
                        raw_chose_data_arr = np.hstack(
                            (raw_chose_data_arr, raw_chose_data))
                # raw_chose_data=np.array(raw_chose_data)
                final_array = np.array(raw_chose_data_arr)
            else:
                for l in range(len(locus_start)):
                    if l == 0:
                        locus_start_index = int(
                            locus_start[l] / space_interval)
                        locus_end_index = int(locus_end[l] / space_interval)
                        raw_chose_data = file['Acquisition/Raw[0]/RawData'][:,
                                                                            locus_start_index:locus_end_index]
                        raw_chose_data_arr = np.array(raw_chose_data)
                    else:
                        locus_start_index = int(
                            locus_start[l] / space_interval)
                        locus_end_index = int(locus_end[l] / space_interval)
                        raw_chose_data = file['Acquisition/Raw[0]/RawData'][:,
                                                                            locus_start_index:locus_end_index]
                        raw_chose_data_arr = np.hstack(
                            (raw_chose_data_arr, raw_chose_data))
                # raw_chose_data=np.array(raw_chose_data)
                final_array = np.vstack((final_array, raw_chose_data_arr))

        file_index += 1
        print(file_index)
        #dimes = data_array.shape
    return final_array, space_interval, FileSampleRate

