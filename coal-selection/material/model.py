import h5py
import numpy as np
from obspy.core import UTCDateTime
from scipy.fft import fft, ifft, fft2
import glob
import os


# Calcuate the weight_list of everychannel #高斯，叠加，归一化，sigma=half_Number
def cal_prob_weight(locus_index_list):
    C_proweight = 0.5
    num_c = locus_index_list[2]-locus_index_list[0]
    num_1 = num_c+1
    num_2 = locus_index_list[1]-locus_index_list[2]+1
    prob_list1 = np.linspace(0, (1-C_proweight)/num_1, num_1)
    prob_list1 = np.delete(prob_list1, [0])
    prob_list1 = np.append(prob_list1, [C_proweight])
    prob_list2 = np.linspace(0, (1-C_proweight)/num_2, num_2)
    prob_list2 = np.delete(prob_list2, [0])
    prob_list2 = prob_list2[::-1]
    prob_list = np.hstack((prob_list1, prob_list2))
    return prob_list


def gaussian(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def cal_pro_weight_use_gaussian(device_poistions, peak_poistions):
    num_peaks = len(peak_poistions)
    length = device_poistions[1]-device_poistions[0]
    sigmas = length/(2*num_peaks)
    # 创建高斯混合分布
    x = np.linspace(device_poistions[0], device_poistions[1], length)
    gmm_pdf = np.zeros_like(x)
    for mu in peak_poistions:
        w = gaussian(x, mu, sigmas)
        gmm_pdf += w

    # 归一化高斯混合分布
    gmm_pdf /= np.sum(gmm_pdf)
    return gmm_pdf


# Standardization
def standardization_row(data):
    data_features_mean = np.mean(data, axis=1)  # 计算每一行的平均值
    data_features_sigma = np.std(data, axis=1)  # 计算每一行的标准差
    a = data_features_sigma.reshape((data.shape[0], 1))
    data_features = (data - data_features_mean.reshape((data.shape[0], 1))) / data_features_sigma.reshape(
        (data.shape[0], 1))  # 标准归一化=(raw-mean)/std
    return data_features

# Read_Data


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

# FFT


def FFT(data, time_length, frequency, target_frequency):
    reshapewindowlength = frequency*time_length
    df = frequency / reshapewindowlength
    pad_delta_time = (data.shape[0]) % (reshapewindowlength)
    signal_array = []
    if pad_delta_time == 0:
        pad_delta_time = 0
    else:
        pad_delta_time = reshapewindowlength-pad_delta_time
    data = np.pad(data, ((0, int(pad_delta_time)), (0, 0)),
                  'constant', constant_values=0)
    # print(data.shape)
    # data=data.reshape(int(data.shape[0]/reshapewindowlength),int(reshapewindowlength))
    for m in range(data.shape[1]):
        procesdata = data[:, m].reshape(
            int(data[:, m].shape[0]/reshapewindowlength), int(reshapewindowlength))
        fft_sample = procesdata
        frequency_data = fft(fft_sample)  # fft axis=-1
        frequency_data_half = np.abs(
            frequency_data[:, :frequency_data.shape[1] // 2])
        #frequency_data_half = standardization(frequency_data_half)
        frequency_data_chose = frequency_data_half[:, :int(
            target_frequency/df)]
        frequency_data_chose = standardization_row(frequency_data_chose)
        signal_array.append(frequency_data_chose)
    signal_array = np.array(signal_array)
    # signal_array=signal_array[:,:2000]
    # df=frequency/reshapewindowlength
    return signal_array, df


'''This above FFT Function is used to caluate the FFT result(abs) for 1d array, 
and it will be sliced into the numbers of small data block which has the setted windowlength'''

# Calculate the correlation coefficient


def cal_corr_coeff(templ_arr, data):
    coefficient = np.abs(np.sum(np.multiply(templ_arr,
                                            data))/np.sqrt(np.sum(templ_arr**2)*np.sum(data**2)))
    return coefficient


'''The above cal_corr_coeff Function is used to caluate the correlation 
coefficient between the template and real_frequency_data'''

# Calculate the Equipment failure probability


def cal_equ_prob(weight_list, coefficient_list):
    prob = 0
    for S in range(weight_list.shape[0]):
        prob += weight_list[S]*coefficient_list[S]
    return prob


'''The above cal_equ_prob Function is used to 
calute the Equipment failure or stable probability
according to the weight & correlation coefficient'''


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
        print("now file index:", file_index)
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


def MyFFT(waveTensor, targetFreq, df):
    specTensor = fft(waveTensor)
    specHalfTensor = np.abs(specTensor[:, :, :specTensor.shape[2] // 2])
    specChoseTensor = specHalfTensor[:, :, :int(targetFreq / df)]

    meanTensor = np.expand_dims(np.mean(specChoseTensor, axis=2), axis=2)
    stdTensor = np.expand_dims(np.std(specChoseTensor, axis=2), axis=2)

    specChoseTensor = (specChoseTensor - meanTensor) / stdTensor

    return specChoseTensor


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

# cch


class Pca_Kmeans_data:
    """
    使用该类之前需要在py文件当前运行的路径下创建几个空文件夹,"PCA_pidaiji"，"Kmeans_pidaiji_stable","Kmeans_pidaiji_static"
    """

    def __init__(self, device_name, loc_start, loc_end):
        self.device_name = device_name
        self.loc_start = loc_start
        self.loc_end = loc_end
        self.Target_Frequency = 1000
        self.Timewindow_Length = 1
        self.Length = 10
        self.n_components = 64
        self.device_path = os.path.join('./PCA_pidaiji/', self.device_name)
        if not os.path.exists(self.device_path):
            print("The path of "+self.device_name+" does not exist.")
            os.makedirs(self.device_path)
            print("The path of "+self.device_name+" has been established.")
        else:
            print("The path of "+self.device_name+" exists.")
        self.pcafig_path = self.device_path + '/' + '_0601_0607_Eigenvalue_Distribution_n_components=' + str(
            self.n_components) + '_' + str(self.Length) + 'overlapbeforePCA.png'
        self.pcamodel_path = self.device_path + '/' + '_0601_0607_Eigenvalue_Distribution_n_components=' + str(
            self.n_components) + '_' + str(self.Length) + 'overlapbeforePCA_function.pkl'
        self.pcalog_path = self.device_path + '/' + '_0601_0607_Eigenvalue_Distribution_n_components=' + str(
            self.n_components) + '_' + str(self.Length) + 'overlapbeforePCA_log.h5'
        self.max_clusters = 8

    def read_process_data(self, path_data, loc_start, loc_end):
        rawDataArray = read_data(path_data, loc_start, loc_end)
        df = 1 / self.Timewindow_Length
        lenWindowIdx = self.Timewindow_Length * rawDataArray[2]
        waveTensor = TimeSegment(rawDataArray[0], lenWindowIdx)
        specTensor = MyFFT(waveTensor, self.Target_Frequency, df)
        final_result = []
        for i in range(specTensor.shape[0]):
            locus_frequency_data = specTensor[i]
            result = []
            for t in range(0, locus_frequency_data.shape[0] - self.Length + 1, self.Length // 2):
                sliced_array = locus_frequency_data[t:t + self.Length]
                merged_array = np.reshape(
                    sliced_array, (-1, self.Length * locus_frequency_data.shape[1]))
                merged_array = (
                    merged_array - np.mean(merged_array)) / np.std(merged_array)
                result.append(merged_array)

            # 将结果转换为一个 numpy 数组
            result = np.vstack(result)
            final_result.append(result)
        specTensor = np.array(final_result)
        self.reshapetensor = specTensor.reshape(
            int(specTensor.shape[0] * specTensor.shape[1]), specTensor.shape[2])
        return self.reshapetensor

    def pca_template(self, data):
        # PCA降维
        pca = PCA(n_components=self.n_components)
        reshapetensor_pca = pca.fit_transform(data)
        # 协方差矩阵
        cov_matritx = pca.get_covariance()
        # 特征值计算
        # lambda_EA = np.abs(np.sort(-np.abs(np.linalg.eigvals(cov_matritx))))

        eigenvalues = pca.explained_variance_

        # 可视化特征值分布
        plt.figure()
        plt.plot(range(len(eigenvalues)), eigenvalues)
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalue Distribution n_components=' +
                  str(self.n_components))
        plt.savefig(self.pcafig_path)
        # plt.show()

        # 保存结果
        pk.dump(pca, open(self.pcamodel_path, "wb"))
        with h5py.File(self.pcalog_path, 'w') as PCAData:
            PCAData.attrs['Original_shape'] = self.reshapetensor.shape
            PCAData.attrs['Reduced_shape'] = reshapetensor_pca.shape
            PrincipalCom_save = PCAData.create_dataset('PCA_Result', reshapetensor_pca.shape,
                                                       dtype=reshapetensor_pca.dtype)
            PrincipalCom_save[:] = reshapetensor_pca
            PrincipalCom_save = PCAData.create_dataset('Principal_components', pca.components_.shape,
                                                       dtype=pca.components_.dtype)
            PrincipalCom_save[:] = pca.components_
            ExplainedVarianceRation_save = PCAData.create_dataset('Explained_variance_ratio',
                                                                  pca.explained_variance_ratio_.shape,
                                                                  dtype=pca.explained_variance_ratio_.dtype)
            ExplainedVarianceRation_save[:] = pca.explained_variance_ratio_
            PCAMeansSave = PCAData.create_dataset('Means', pca.mean_.shape,
                                                  dtype=pca.mean_.dtype)
            PCAMeansSave[:] = pca.mean_
            PCAeigenvaluesSave = PCAData.create_dataset('Eigenvalues', pca.explained_variance_.shape,
                                                        dtype=pca.explained_variance_.dtype)
            PCAeigenvaluesSave[:] = pca.explained_variance_
            PCAsingularvaluesSave = PCAData.create_dataset('Singularvalues', pca.singular_values_.shape,
                                                           dtype=pca.singular_values_.dtype)
            PCAsingularvaluesSave[:] = pca.singular_values_

    def kmeans_template(self, path_kmeans):
        if path_kmeans.find("stable") != -1:
            state = "stable"
        else:
            state = "static"
        device_kmeans_path = os.path.join(
            './Kmeans_pidaiji_' + state, self.device_path)
        if not os.path.exists(device_kmeans_path):
            print("The path of " + self.device_name +
                  " does not exist for kmeans.")
            os.makedirs(device_kmeans_path)
            print("The path of " + self.device_name +
                  " has been established for kmeans.")
        else:
            print("The path of " + self.device_name + " exists for kmeans.")
        reshapetensor = self.read_process_data(
            path_kmeans, self.loc_start, self.loc_end)
        pca_reload = pk.load(open(self.pcamodel_path, 'rb'))
        PCAResult = pca_reload.transform(reshapetensor)
        PCAResult = standardization_row(PCAResult)

        wcss = []
        centre_list = []
        score_list = []

        # 遍历不同的聚类数
        for i in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(PCAResult)
            kmeans_labels = kmeans.fit_predict(PCAResult)
            if i != 1:
                score = silhouette_score(PCAResult, kmeans_labels)
                score_list.append(score)
            else:
                score_list.append(0)
            wcss.append(kmeans.inertia_)
            centre_list.append(kmeans.cluster_centers_)
            cluster_labels = kmeans.labels_
            pca = PCA(n_components=2)
            spectra_data_2d = pca.fit_transform(PCAResult)
            plt.figure()
            plt.scatter(spectra_data_2d[:, 0], spectra_data_2d[:, 1], c=cluster_labels, cmap='viridis', marker='o',
                        s=50)
            plt.title(
                self.device_name + '_0601_0607_overlapbeforepca_'+state+'_K-means Clustering Visualization_n_components=' + str(
                    self.n_components) + '_Length=' + str(self.Length) + '_' + str(i))
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.colorbar().set_label('Cluster Label')
            # plt.show()
            plt.savefig(
                device_kmeans_path+'/' + '_0601_0607_overlapbeforepca_'+'_means_n_components=' + str(
                    self.n_components) + '_Length=' + str(self.Length) + '_' + str(i) + '.png')

        # 绘制手肘图
        plt.figure()
        plt.plot(range(1, self.max_clusters + 1), wcss, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.title(self.device_name + '_0601_0607_overlapbeforepca_'+state+'_means_n_components=' + str(
            self.n_components) + '_Elbow Method')
        # plt.show()
        plt.savefig(
            device_kmeans_path+'/' + '_0601_0607_overlapbeforepca_means_n_components=' + str(
                self.n_components) + '_Length=' + str(self.Length) + '_Elbow Method.png')
        plt.figure()
        plt.plot(range(1, self.max_clusters + 1), score_list, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('score')
        plt.title(self.device_name + 'Score Method')
        plt.savefig(
            device_kmeans_path+'/' + '_0601_0607_overlapbeforepca_means_n_components=' + str(
                self.n_components) + '_Length=' + str(self.Length) + '_score_method.png')
        with h5py.File(device_kmeans_path+'/' + '_0601_0607_overlapbeforepca_means_n_components=' + str(
                self.n_components) + '_PCA_means_template_dataset_' + 'Length=' + str(self.Length) + '_log.h5',
                'w') as Temp:
            for knum in range(len(centre_list)):
                Temp_save = Temp.create_dataset('Template_K=' + str(knum + 1), centre_list[knum].shape,
                                                dtype=centre_list[knum].dtype)
                Temp_save[:] = centre_list[knum]
