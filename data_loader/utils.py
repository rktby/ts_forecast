import numpy as np
from tensorflow.data import Dataset 

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
        d = np.array(dataset[field['name']])
        d = norm(d, normalise=field['normalise'])
        d_vals.append(conv(d[0], length, start=start, stop=stop, stride=stride, dim=dim))
        d_mins.append(conv(d[1], length, start=start, stop=stop, stride=stride, dim=dim))
        d_maxs.append(conv(d[2], length, start=start, stop=stop, stride=stride, dim=dim))
    
    return np.concatenate(d_vals, axis=-1), np.concatenate(d_mins, axis=-1), np.concatenate(d_maxs, axis=-1)

def split_data(hparams, data):
    """
    Break a long time series dataset down into shorter frames for training

    Arguments
    ---------
    hparams: dict
        Dictionary defining the train, validation and test split points
        hparams.train_split: Percentage of the dataset to put in train dataset
        hparams.val_split:   Percentage of the dataset to put in validation dataset
        hparams.test_split:  Percentage of the dataset to put in test dataset
        hparams.batch_size:  Size of training batches
    data: list[dataframe_1, dataframe_2, ..., dataframe_n]
        List of datasets to return in training batches

    Returns
    -------
    data_train: tf.data.Dataset
        Training dataset of batch size hparams.batch_size
    data_val:   tf.data.Dataset
        Validation dataset of batch size hparams.batch_size
    data_test:  tf.data.Dataset
        Test dataset of batch size hparams.batch_size
    """
    # Calculate train, validate and test set sizes
    p_train, p_val, p_test = hparams.train_split, hparams.val_split, hparams.test_split
    assert(sum([p_train, p_val, p_test]) == 1)

    train_pos = int(data[0].shape[0] * p_train)
    val_pos   = int(data[0].shape[0] * (p_train + p_val))
    end_pos   = int(data[0].shape[0])
    
    # Split dataset into training, validation and test sets
    def _split_(data, start, finish):
        data = tuple(d[start:finish].astype(np.float32) for d in data)
        data = Dataset.from_tensor_slices(data)
        
        return data.batch(hparams.batch_size, drop_remainder=True)
    
    data_train = _split_(data, 0, train_pos)
    data_val   = _split_(data, train_pos, val_pos)
    data_test  = _split_(data, val_pos, end_pos)
    
    return data_train, data_val, data_test

def norm(dataset, normalise='global_max', norm_epsilon=1e-12):
    """
    Normalise dataset on scale [0..1]
    
    Parameters
    ----------
    dataset : ndarray
        dataset to normalise
    normalise : string or int
        The axis and method for normalising the data
        -'global_max' or 0: Scale all observations and down by max across the dataset
        -'local_max' or 1:  Scale all observations and down by max row-wise
        -'global_max_min' or 2: Subtract min from all observations and scale down by (max-min) across the dataset
        -'local_max_min' or 3:  Subtract min from all observations and scale down by (max-min) row-wise
    
    Returns
    -------
    datanorm : ndarray
        The normalised dataset
    x_max : ndarray
        The maximum values - used to rescale dataset
    x_min : ndarray
        The minimum values - used to rescale dataset
    """
    # Ensure dataset is at least 3D 
    if len(dataset.shape) == 1:
        dataset = np.expand_dims(dataset, 0)
    if len(dataset.shape) == 2:
        dataset = np.expand_dims(dataset, -1)
    
    # 
    if isinstance(normalise, int):
        if   normalise == 0: normalise = 'global_max'
        elif normalise == 1: normalise = 'local_max'
        elif normalise == 2: normalise = 'global_max_min'
        elif normalise == 3: normalise = 'local_max_min'
        else: normalise = None
    
    # Set axis to perform normalise calculation on
    norm_axis = (0,1) if normalise.count('global') == 1 else 1
    
    # Set max normalisation factor
    if normalise.count('max') > 0:
        x_max = np.max(dataset, axis=norm_axis, keepdims=True)
        x_max[x_max == 0] = norm_epsilon
        x_max = x_max * np.ones_like(dataset)
    else:
        x_max = np.ones_like(dataset)
        
    # Set min normalisation factor
    if normalise.count('min') > 0:
        x_min = np.min(dataset, axis=norm_axis, keepdims=True)
        x_min = x_min + np.zeros_like(dataset)
    else:
        x_min = np.zeros_like(dataset)
    
    # Normalise dataset
    datanorm = (dataset - x_min) / (x_max - x_min)
    
    return datanorm, x_max, x_min

def conv(dataset, length, start=0, stop=-1, stride=1, dim=1):
    """
    Split dataset into segments
    
    Parameters
    ----------
    dataset: ndarray
        Dataset to split
    length: int
        Length of each segment
    start: int
        Index in dataset to start at (default = 0)
    step: int
        Step size to take between segments (default = 1)
    dim: int
        Dimensions to wrap dataset around into
        
    Returns
    -------
    dataset: ndarray
        Reshaped dataset split into segments
    """
    assert length % dim == 0
    
    if len(dataset.shape) == 1:
        dataset = np.expand_dims(dataset, 0)
    if len(dataset.shape) == 2:
        dataset = np.expand_dims(dataset, -1)
    
    nx, ny, nz = dataset[:,:stop,:].shape
    ix = (ny - length + stride) % stride
    
    dataset = [dataset[i,j:j+length] for j in range(start, ny-ix-length+1, stride) for i in range(nx)]
    
    #sx, sy, sz = dataset.strides
    
    #ny = ny - length + step
    #nx = nx * (ny // step)
    #ny = length
    #sx = sz * step
    
    #np.lib.stride_tricks.as_strided(c[:,:-ix], shape=(nx, ny, nz), strides=(sx,sy,sz)).squeeze()
    
    # Reshape dataset
    dataset = np.array(dataset)
    dataset = dataset.reshape(-1,length,1)
    if dim > 1:
        dataset = np.reshape(dataset, (len(dataset), dim, -1))
        dataset = np.transpose(dataset, (0,2,1))
    
    return dataset
