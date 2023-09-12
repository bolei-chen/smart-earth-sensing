import h5py
import numpy as np
import glob

def read_data(path, locus_start, locus_end):
    files = sorted(glob.glob(path))
    file_index = 0
    raw_chose_data_arr = []
    final_array = []
    FileSampleRate = 0
    space_interval = 0
    for file in files:
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
