import tensorflow as tf
import tensorflow.keras as tfk

class CEEModel(tf.keras.Model):
    def __init__(self, hparams):
        super(CEEModel, self).__init__()
        self.units = hparams.units
        self.dropout_rate = hparams.dropout_rate
        self.training = True
        
        # Store processing instructions for embedding input, conditioning and output variables
        self.inp =  hparams.inp
        self.target = hparams.target
        self.cond =   hparams.cond
        
        # Create weight embeddings
        self.we, self.channels = {}, {}
        self.mmult   = tfk.backend.dot

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
    
    def call(self, x, c):
        """
        Default behavious is to embed and then unembed x
        
        Parameters
        ----------
        x: Input variables
            [m x n_in  x d_in]  tensor 
        c: Conditioning variables
            [m x n_out x d_out] tensor 

        Returns
        -------
        x: Predicted variables
            [m x n_out x d_out] tensor
        """

        x = self.embed(x, self.inp['processing'])
        x = self.out(x, self.target['processing'])

        return x