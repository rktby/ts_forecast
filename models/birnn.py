import tensorflow as tf
import tensorflow.keras as tfk

class BiRNN(tf.keras.Model):
    def __init__(self, hparams):
        super(BiRNN, self).__init__()
        self.units = hparams.units
        self.dropout_rate = hparams.dropout_rate
        
        self.inp =  hparams.inp
        self.target = hparams.target
        self.cond =   hparams.cond

        self.output_dim = sum([f['embedding'] for f in self.target['fields']]) * self.target['dim']
        self.training = True
        
        # Create weight embeddings
        self.we, self.channels = {}, {}
        for var in (self.inp, self.target, self.cond):
            var['processing'] = []
            for field in var['fields']:
                name, embedding = field['name'], field['embedding']
                var['processing'] += [(name, embedding > 1)] * var['dim']
                if name not in self.we.keys():
                    self.channels[name] = embedding
                    embedding = 1 if embedding == 1 else embedding + 1
                    self.we[name] = tf.get_variable('embedding_' + name, [embedding, self.units],
                                        initializer=tf.random_normal_initializer(stddev=0.02))

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
        self.mmult   = tfk.backend.dot

    def embed(self, x, methods):
        e = []
        for r, (field, to_embed) in zip(tf.unstack(x, axis=-1), methods):
            if to_embed:
                r = r * self.channels[field]
                r = tf.cast(r, tf.int32)
                r = tf.gather(self.we[field], r)
                e.append(r)
            else:
                r = tf.expand_dims(r, axis=-1)
                e.append(r)

        return tf.concat(e, axis=-1)

    def out(self, x, methods):
        e = []
        for field, to_embed in methods:
            r = self.mmult(x, tf.transpose(self.we[field]))
            e.append(r)
        
        return e
    
    def call(self, x, hidden):
        
        #in_dim = self.inp['dim']
        #out = x[:,-1:,in_dim-1::in_dim]
        
        x = self.embed(x, self.inp['processing'])
        
        x, states, _ = self.gru(x, initial_state=self.reset_states())
        
        x = self.fc_out(self.dropout(x, training=self.training))
        
        x = self.out(x, self.target['processing'])
        
        return x