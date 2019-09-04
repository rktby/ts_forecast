import numpy as np

from .utils import process_data, split_data

def load_data(hparams, mode='fixed_frequency', normalise='fixed_scale',
              add_time_embedding=False, offset_time_embedding = 0, shuffle=False, isVerbose=True):
    """
    Arguments:
        hparams: tf hyperparameters
        mode:
            -'fixed_frequency': sin wave frequency is 1 / hparams.in_seq_len
            -'random_frequency': sin wave frequency is in range 3*hparams.in_seq_len to 0.3*hparams.in_seq_len
        normalise:
            -'fixed_scale': Returns sin curves with amplitude 2
            -'random_scale': Randomly scale curve to have amplitude between 0.2 and 2
            -'random_scale_and_offset': Randomly scale curve to have amplitude between 0.2 and 2
                                       Randomly offset curve so min >= 0 and max <= 2
    """ 
    # Calculate train, validate and test set sizes
    p_test = hparams.test_split
    n_obs = int(hparams.batch_size / p_test)
    
    timescale = np.mgrid[0:n_obs,0:10*hparams.inp['length']][1]
    h_offset  = np.random.randn(n_obs, 1) * 10
    h_offset[0] = 0
    if mode == 'random_frequency':
        frequency = hparams.inp['length'] * np.random.uniform(1/3, 3, (n_obs, 1))
    else:
        frequency = np.full((n_obs, 1), hparams.inp['length'])

    dataset = (timescale + h_offset) * 2 * np.pi / frequency
    dataset = np.sin(dataset) + 1
    
    if normalise.find('random_scale') >= 0:
        divisor = np.random.uniform(1, 10, (n_obs, 1))
        dataset /= divisor
        
    if normalise == 'random_scale_and_offset':
        offset = np.random.uniform(0, 2 - 2 / divisor, (n_obs, 1))
        dataset += offset
    
    mask = np.ones_like(dataset)
    
    if add_time_embedding:
        timescale = timescale % frequency + offset_time_embedding
        mask = np.dstack((mask, timescale))
    
    dataset = {'sin': dataset}
    # Split into input, target and conditioning datasets
    inp,    inp_max,    inp_min    = process_data(hparams.inp, dataset)
    target, target_max, target_min = process_data(hparams.target, dataset)
    
    # Split into training, validation and test datasets
    train, val, test = split_data(hparams, [inp, inp_max, inp_min, target, target_max, target_min, mask])

    return train, val, test