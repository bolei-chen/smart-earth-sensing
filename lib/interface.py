from h5reader import * 
from utils import * 
from dataset import * 

''' 
notice this function only handles 1 h5 file with 1 locus id at a time 
the sampling rate must be 1k Hz otherwise the model will not longer be accurate

intput: 
    h5_path: path to the h5 file
        notice this function only handles only one h5 file with one locus id at a time
        the sampling rate must be 1k Hz otherwise the model will be no longer accurate 
         
    locus_id: 道号 
     
    duration: how many second does the h5 file record 
     
    state_dict_path: the path of the stored model state dict
        notice the model only takes in data of shape (batch_size, 1, 26, 64) 
        for a single piece of data, just set batch_size=1 
         
output:
    final_pred: the prediction made by the model for this period of time 
''' 
def predict(h5_path, locus_id, model, sr):
    raw, _, _ = read_data(h5_path, [locus_id], [locus_id + 1]) 
    features = np.array(list(div2chunks(raw, sr / 2)))
    features = np.array([raw2mfcc(raw) for raw in features])
    features = np.array([[sample] for sample in features]) 
    (l, _, m, n) = np.shape(features) 
    xs = torch.reshape(torch.tensor(features, dtype=torch.float32), (l, 1, m, n))
    y_hats = model(xs)
    y_hats = [torch.argmax(y_hat) for y_hat in y_hats]
    y_hat = int(max(y_hats, y_hats.count)) 
    return y_hat 

def main():
    n = len(sys.argv) 
    if n != 4:
        print('invalid number of inputs, should be 3') 
        return 
    h5_path = sys.argv[1]
    locus_id = int(sys.argv[2])
    state_dict_path = sys.argv[3] 
    predict(h5_path, locus_id, state_dict_path) 

     
     
     
