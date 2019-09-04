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