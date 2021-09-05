"""Data Loader using tfio"""
# external
import jsonschema
import tensorflow as tf
import tensorflow_io as tfio
from pathlib import Path
import pandas as pd
import numpy as np


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data_from_csv(data_config):
        """ Loads dataset from csv using pandas DataFrames
            Args:
                data_config (json): data configuration parameters
            Returns:
                audio_ds (tf.data.Dataset): audio tf dataset
                label_ds (tf.data.Dataset): categorised annotations tf dataset
                class_vocab (list): annotations vocabulary
                (also saves the class vocabulary into memory)
        """
        # Read csv containing file paths and lables and load it into a pandas DataFrame
        df = pd.read_csv(data_config.metadata_path)
        # Get audio paths --> tensorflow dataset
        file_path_ds = tf.data.Dataset.from_tensor_slices((data_config.audio_dir+df[data_config.paths_col]).values)
        # Load the audio from those paths --> tensorflow dataset
        audio_ds = file_path_ds.map(DataLoader._load_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Categorise the annotations --> codes:[int], class_vocab:[string]
        codes, class_vocab = pd.factorize(df[data_config.label_col])
        # Save the mapping int(index)<->string(class_vocab) as an numpy array
        np.save(data_config.vocab_path+'class_vocab.npy', class_vocab)
        # Get the annotations --> tensor flow dataset
        label_ds = tf.data.Dataset.from_tensor_slices(codes)
        return(audio_ds, label_ds, class_vocab)


    @staticmethod
    def load_audio_from_folder(data_config):
        """ Loads audio dataset from a given directory
            Args:
                data_config (json): data configuration parameters
            Returns:
                audio_ds (tf.data.Dataset): audio tf dataset
        """
        #  Get the audio directory and convert it into a fancy path easier to handle than a string
        audio_root = Path(data_config.audio_dir)
        # Get all the files that comply with the folder pattern --> tensorflow dataset
        file_path_ds = tf.data.Dataset.list_files(str(audio_root/data_config.folder_pattern), shuffle=data_config.shuffle)
        # Load the audio from those paths --> tensorflow dataset
        audio_ds = file_path_ds.map(DataLoader._load_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return audio_ds


    @staticmethod
    def merge_and_split(audio_ds, label_ds, data_config):
        """ Merges audio and labels datasets and splits them into training, testing and validation
            Args:
                audio_ds (tf.data.Dataset): audio tf dataset
                label_ds (tf.data.Dataset): annotations tf dataset
                data_config (json): data configuration parameters
            Returns:
                train_ds (tf.data.Dataset): training (audio,label) tuples
                test_ds (tf.data.Dataset): testing (audio,label) tuples
                val_ds (tf.data.Dataset): validation (audio,label) tuples
        """
        # Merge the audio and labels into a touple tf dataset
        dataset = tf.data.Dataset.zip((audio_ds, label_ds))
        # Calculate the sizes of the train and test datasets (validation taken as leftover)
        ds_size = dataset.cardinality().numpy()
        train_size = int(data_config.train_ratio * ds_size)
        test_size = int(data_config.test_ratio * ds_size)
        # Shuffle the dataset before splitting
        full_ds = dataset.shuffle(100, reshuffle_each_iteration=True)
        # Take the first chunk as training data
        train_ds = full_ds.take(train_size)
        # Take the following chunk as test data
        test_ds = full_ds.skip(train_size)
        # Take the remaining chunk as validation data
        val_ds = test_ds.skip(test_size)
        test_ds = test_ds.take(test_size)
        # Print the number of samples in each partition
        print('* Dataset size : ', ds_size)
        print('--> * Training : ', train_ds.cardinality().numpy())
        print('--> * Validation  : ', val_ds.cardinality().numpy())
        print('--> * Testing  : ', test_ds.cardinality().numpy())
        return train_ds, test_ds, val_ds


    @staticmethod
    def set_batch(train, test, val, batch_size, buffer_size):
        """ Batch the training, testing and validation - with prefetching for training for speed-up
            Args:
                train_ds (tf.data.Dataset): training (audio,label) tuples
                test_ds (tf.data.Dataset): testing (audio,label) tuples
                val_ds (tf.data.Dataset): validation (audio,label) tuples
                batch_size: number of samples in each batch
                buffer_size: number of samples in the buffer for pre-fetching
            Returns:
                train_ds (tf.data.Dataset): batches of training (audio,label) tuples
                test_ds (tf.data.Dataset): batches of testing (audio,label) tuples
                val_ds (tf.data.Dataset): batches of validation (audio,label) tuples
        """
        train_ds = train.shuffle(buffer_size).batch(batch_size)#.cache()#.repeat()
        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_ds = test.batch(batch_size)
        val_ds = val.batch(batch_size)
        return train_ds, test_ds, val_ds


    @staticmethod
    def preprocess_labels(label_ds, class_vocab):
        """ Substitute in categorical label dataset for one-hot vectors dataset (int -> one-hot vector) """
        return label_ds.map(lambda label_code:DataLoader._get_class_one_hot(label_code, class_vocab))


    @staticmethod
    def mlp_preprocess_audio(audio_ds, data_config):
        """ Audio transformations to create the required audio input for the MLP model
            Args:
                audio_ds (tf.data.Dataset): audio tf dataset
                data_config (json): data configuration parameters
            Returns:
                spec_flat_ds (tf.data.Dataset): Stacked 1D Spectrogram dataset of the mono, resampled and choped audio_ds

        """
        # Convert all audio signals in the dataset to mono (i.e. one channel):
        mono_ds = audio_ds.map(lambda audio,rate:(DataLoader._tomono(audio),rate))
        # Resample mono audio signals to given sample rate
        resample_ds = mono_ds.map(lambda audio,rate:DataLoader._resample(audio, rate_in=tf.cast(rate,tf.int64), rate_out=data_config.sample_rate))
        # Chop audio to a specific duration
        chop_ds = resample_ds.map(lambda audio:DataLoader._chop_duration(audio,data_config.sample_rate,data_config.duration_sec) )
        # Compute spectrogram
        spec_ds = chop_ds.map(lambda audio:DataLoader._tospectrogram(audio, data_config.nfft, data_config.win_size, data_config.hop_size))
        # Flatten the spectrogram (because dummy model only takes 1D)
        spec_flat_ds = spec_ds.map(lambda X:tf.reshape(X,[-1]))
        return spec_flat_ds


    @staticmethod
    def _get_class_one_hot(label_code, class_vocab):
        return tf.one_hot(label_code, depth= len(class_vocab))


    @staticmethod
    def _load_audio(file_path):
        audio = tfio.IOTensor.graph(tf.float32).from_audio(file_path)
        return audio.to_tensor(), audio.rate

    @staticmethod
    def _tomono(audio_tensor):
        return tf.math.reduce_mean(audio_tensor, axis=1)

    @staticmethod
    def _resample(audio_tensor, rate_in, rate_out):
        return tfio.audio.resample(audio_tensor, rate_in=rate_in, rate_out=rate_out)

    @staticmethod
    def _chop_duration(audio_tensor, audio_rate, duration_sec):
        return audio_tensor[:audio_rate*duration_sec]

    @staticmethod
    def _tospectrogram(audio_tensor, nfft, win_size, hop_size ):
        #transpose so the dimensions match (frequency, time)
        return tf.transpose(tfio.experimental.audio.spectrogram(audio_tensor, nfft=nfft, window=win_size, stride=hop_size))
