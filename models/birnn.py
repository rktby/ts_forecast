import tensorflow as tf
import tensorflow.keras as tfk
from .cee_model import CEEModel

class BiRNN(CEEModel):
    def __init__(self, hparams):
        super(BiRNN, self).__init__(hparams)
        # Create GRU Cell
        if tf.test.is_gpu_available():
            self.gru = tfk.layers.Bidirectional(\
                            tfk.layers.CuDNNGRU(self.units, return_sequences=True, return_state=True, name='gru'))
        else:
            self.gru = tfk.layers.Bidirectional(\
                            tfk.layers.GRU(self.units, return_sequences=True, return_state=True, 
                                                   recurrent_activation='relu', name='gru'))

        self.dropout = tfk.layers.Dropout(self.dropout_rate)
        self.fc_out  = tfk.layers.Dense(self.units, activation='linear', name='affine_out')

    def call(self, x, hidden):
        
        x = self.embed(x, self.inp['processing'])
        
        x, states, _ = self.gru(x, initial_state=self.reset_states())
        
        x = self.fc_out(self.dropout(x, training=self.training))
        
        x = self.out(x, self.target['processing'])
        
        return x