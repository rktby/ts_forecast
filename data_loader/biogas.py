import numpy as np
import pandas as pd
from .utils import *

def load_data(hparams, shuffle=False):
    """
    Load data from the "biogas" dataset
    
    Parameters
    ----------
    hparams : hyperparameter object
        Contains frames for hparams.inp, hparams.target, hparams.cond
        {
        'fields': [['field', 'cont' | 'cat', normalise method (0-4)], ...],
        'start': int, 'stop': int, 'length': int, 'stride': int, 'dim': int
        }
        hparams.val_split
        hparams.test_split
    normalise: string
        {'global_max', 'global_min_max', 'local_max', 'local_min_max'}
    shuffle: boolean
    
    Returns
    -------
    train: tf.data.Dataset
        Training dataset of batch size hparams.batch_size
        A batch contains [input, input_max, input_min, target, target_max, target_min, conditioning]
    val:   tf.data.Dataset
        Validation dataset of batch size hparams.batch_size
        A batch contains [input, input_max, input_min, target, target_max, target_min, conditioning]
    test:  tf.data.Dataset
        Test dataset of batch size hparams.batch_size
        A batch contains [input, input_max, input_min, target, target_max, target_min, conditioning]
    """ 
    # Load dataset
    dataset = pd.read_csv('../../Data/cr2c_opdata_TMP_PRESSURE_TEMP_WATER_COND_GAS_PH_DPI_LEVEL.csv')[:10024]
    calib = pd.read_csv('../../Data/calibration_data.csv')
    
    dataset['Date'] = pd.to_datetime(dataset['Time']).dt.date
    calib['Date']   = pd.to_datetime(calib['Date']).dt.date
    
    dataset = pd.merge(calib[['Date', 'Error_bin']], dataset, how='inner', on='Date')
    del(calib)
    
    cond_fields = [field['name'] for field in hparams.cond['fields']]
    
    # Split into input, target and conditioning datasets
    inp,    inp_max,    inp_min    = process_data(hparams.inp, dataset)
    target, target_max, target_min = process_data(hparams.target, dataset)
    cond                           = np.isfinite(dataset[cond_fields].values).reshape(-1,1,1)
    
    # Split into training, validation and test datasets
    train, val, test = split_data(hparams, [inp, inp_max, inp_min, target, target_max, target_min, cond])
    
    return train, val, test

def process_data(frame, dataset):
    """
    Break a long time series dataset down into shorter frames for training
    
    Parameters
    ----------
    frame: dict
        Dictionary defining the parameters of the frame
        frame['start']:  Position in the dataset to start taking data from
        frame['stop']:   Position in the dataset to stop taking data from
        frame['length']: Length of the frame to take from the dataset
        frame['stride']: Size of the stride to take between each frame
        frame['dim']:    Number of dimensions to wrap the data to
                         Must be a whole factor of 'length'
                         e.g. ['length': 120, 'dim': 5] => (24 x 5) frame
        frame['fields']: List of the fields in the dataset to process
    normalise: str
        Normalisation method - see utils.norm for documentation
    dataset: dataframe
        Dataset
    
    Returns
    -------
    d_vals: dataframe
        Normalised values from dataset
    d_mins: dataframe
        Minimums used to normalise dataset
    d_maxs: dataframe
        Maximums used to normalise dataset
    """
    d_vals, d_mins, d_maxs = [], [], []
    start, stop, length, stride, dim = frame['start'], frame['stop'], frame['length'], frame['stride'], frame['dim']
    assert(length % dim == 0)
    
    for field in frame['fields']:
        d = dataset[field['name']].values
        d = norm(d, normalise=field['normalise'])
        d_vals.append(conv(d[0], length, start=start, stop=stop, stride=stride, dim=dim))
        d_mins.append(conv(d[1], length, start=start, stop=stop, stride=stride, dim=dim))
        d_maxs.append(conv(d[2], length, start=start, stop=stop, stride=stride, dim=dim))
    
    return np.concatenate(d_vals, axis=-1), np.concatenate(d_mins, axis=-1), np.concatenate(d_maxs, axis=-1)