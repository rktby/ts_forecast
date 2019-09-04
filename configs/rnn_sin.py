class hparams():
    def __init__(self):

        self.inp = {
            'fields': [{'name': 'sin', 'embedding': 1, 'normalise': 2}],
            'start': 0, 'stop': 120, 'length': 120, 'stride': 1, 'dim': 5
        }
        self.target = {
            'fields': [{'name': 'sin', 'embedding': 1, 'normalise': 2}],
            'start': 120, 'stop': 144, 'length': 24, 'stride': 1, 'dim': 1
        }
        self.cond = {
            'fields': [{'name': 'AT305', 'embedding': 1, 'normalise': 2}],
            'start': 120, 'stop': 10024, 'length': 24, 'stride': 1, 'dim': 1
        }

        self.batch_size = 800
        self.datagen = 'biogas'
        self.lambd = 1e-06
        self.learning_rate = 0.01
        self.logs_path = '/tmp/tensorflow_logs'
        self.lr_decay = 0.999
        self.units = 32
        self.norm_epsilon = 1e-12
        self.num_layers = 1
        self.dropout_rate = 0.2

        self.test_split = 0.1
        self.train_split = 0.8
        self.val_split = 0.1
