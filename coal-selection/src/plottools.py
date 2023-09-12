import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
 
def generate_plot(h5path, device_id, device_name, plot_path): 
    h5s = np.sort(glob.glob(h5path)) 
    device_result_list = []
    sensor_result_list = []
    soft_max_result_list = []
     
    for h5 in h5s:
        h5 = h5py.File(h5, 'r')
        device_result_data = h5[device_id + 'device_results']
        sensor_result_data = h5[device_id + 'sensor_results']
        soft_max_result_data = h5[device_id + 'soft_max_results']
        device_result_list.append(device_result_data)
        sensor_result_list.append(sensor_result_data)
        soft_max_result_list.append(soft_max_result_data)
    device_result_list = np.hstack(device_result_list)
    soft_max_result_list = np.vstack(soft_max_result_list)
    sensor_result_list = np.concatenate(tuple(sensor_result_list), axis=1)
    stable_max = np.max([result[0] for result in device_result_list]) 
    stable_min = np.min([result[0] for result in device_result_list]) 
    static_max = np.max([result[1] for result in device_result_list]) 
    static_min = np.min([result[1] for result in device_result_list]) 
     
    plt.figure()
    plt.title(device_id + 'Device_Distance_Result v.s. Time')
    plt.plot(device_result_list[0],
             label='Distance from Stable Template', color='orange')
    plt.plot(device_result_list[1],
             label='Distance from Static Template', color='blue')
    plt.xlabel('Time (Second)')
    plt.ylabel('Distance')
    plt.legend()
    plt.savefig(plot_path + str(device_id) + '_' + str(device_name) + '.png')
    plt.close() 
    return stable_max, stable_min, static_max, stable_min 
