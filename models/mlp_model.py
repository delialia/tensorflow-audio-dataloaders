''' Dummy model to test dataloders'''
#external
import tensorflow as tf
import tensorflow_datasets as tfds

#internal
from utils.config import Config
from .base_model import BaseModel
from dataloaders.dataloader_tfio import DataLoader

class MLPModel(BaseModel):
    """MLP Model inheriting a Base Model"""

    def __init__(self, cfg, name='mlp-basic'):
        self.config = Config.from_json(cfg)
        self.name = name
        self.dataset = None
        self.datasetinfo = None
        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epoches = self.config.train.epoches
        self.steps_per_epoch = 0
        self.validation_steps = 0
        self.val_subsplits = self.config.train.val_subsplits
        self.feature_dim = 0
        self.model = None
        self.model_path = self.config.train.model_path
        self.num_classes = None


    def load_data(self, method):
        """Loads and Preprocess data  either using tfio or tfds passed in method"""
        if method == 'tfio':
            # Get audio paths, their labels and vocabulary
            audio_ds, label_ds, class_vocab = DataLoader().load_data_from_csv(self.config.data)
            # Set number of classes
            self.num_classes = len(class_vocab)
            # Convert labels to one-hot vectors
            label_ds = DataLoader.preprocess_labels(label_ds, class_vocab)
            # Preprocess audio for this model
            audio_ds = DataLoader.mlp_preprocess_audio(audio_ds, self.config.data)
            # Get the feature space dimension:
            for item in audio_ds.take(1):
                self.feature_dim = item.get_shape()
                print('* Feature Space DIM:', item.get_shape())
            # Merge audio and labels into one dataset and split them to training, testing and validaton datasets
            train_ds, test_ds, val_ds = DataLoader.merge_and_split(audio_ds, label_ds, self.config.data)
            # Prepare training, testing and validation batches
            train_ds, test_ds, val_ds = DataLoader.set_batch(train_ds, test_ds, val_ds, self.batch_size, self.buffer_size)
            # Allocate class dataset attribute
            self.dataset = {'train':None, 'test':None, 'val':None}
            self.dataset['train'] = train_ds
            self.dataset['test'] = test_ds
            self.dataset['val'] = val_ds

        elif method == 'tfds':
            self.dataset, self.datasetinfo = tfds.load('test_dataset',
                                                with_info=True,
                                                as_supervised = True, # this will get whatever tuple defined as supervised in my_tfds.py
                                                batch_size = self.batch_size,
                                                shuffle_files = True)
            self.num_classes = self.datasetinfo.features['label'].num_classes
            self.feature_dim = self.datasetinfo.features['audio'].shape
            for split in self.dataset:
                self.dataset[split] = self.dataset[split].map(lambda audio,label:(audio,tf.one_hot(label,depth=self.num_classes)))

        # Set training parameters
        self._set_training_parameters()



    def build(self):
        """ Builds MLP with Keras """
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=self.feature_dim))
        self.model.add(tf.keras.layers.Dropout(.8))
        self.model.add(tf.keras.layers.Dense(1024,activation='relu'))
        self.model.add(tf.keras.layers.Dense(512,activation='relu'))
        self.model.add(tf.keras.layers.Dense(128,activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        print('* Keras Model was built successfully')


    def train(self):
        """Compiles and trains the model"""
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.config.train.optimizer.type, #'adam',
                      metrics=self.config.train.metrics )#['accuracy']
        self.model.fit(self.dataset['train'],
                        epochs=self.epoches,
                        steps_per_epoch=self.steps_per_epoch,
                        validation_steps=self.validation_steps,
                        validation_data=self.dataset['val'],
                        verbose=1)

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        test_results = self.model.evaluate(self.dataset['test'], verbose=1)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')


    def save(self):
        '''Saves the trained model'''
        self.model.save(self.model_path+self.name)


    def _set_training_parameters(self):
        """Sets training parameters"""
        # How many training batches to consider per epoch? Set to all.
        self.steps_per_epoch = self.dataset['train'].cardinality().numpy() #OR self.train_length // self.batch_size
        # How many validation batches to consider per epoch?
        self.validation_steps = self.dataset['val'].cardinality().numpy() // self.val_subsplits #OR self.val_length // self.batch_size // self.val_subsplits
        # Print out the training parameters
        print('* Training parameters :')
        print(' -> Batch size : ',self.batch_size)
        print(' -> Steps per epoch : ', self.steps_per_epoch )
        print(' -> Val subsplits : ', self.val_subsplits )
        print(' -> Val steps per epoch : ', self.validation_steps )
