"""Data Loader using tfds
NOTE : I had to > set ulimit -n 4096  on my macOS Catalina 10.15.7
******************************************************************************************
There are a couple of ways of building the dataset you define here.
The most straight forward one for me was to type in the terminal :
> tfds build
Check out the flags: https://www.tensorflow.org/datasets/cli
Eg: if you're just testing:  '--max_examples_per_split'  can be useful
Unless otherwise specified, this will dump your dataset in ~/tensorflow_datasets
in TFRecords format with some nice info .json files
******************************************************************************************
"""
#external
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
# for handling the csv file containing data info:
import pandas as pd
# you will only need this if you want to preprocess audio (you could also use librosa):
import tensorflow_io as tfio


# >>>>>>>>>> CHANGE THIS WITH YOUR INFO <<<<<<<<<<<<<<<<<<
# Set Configuration
_DESCRIPTION = """
This is a description of your dataset
"""
# _MANUAL_DOWNLOAD_INSTRUCTIONS = """
# You can also put here instructions to download the dataset """
_AUDIODIR = '/path/to/audio/mp3/'
_METADATAPATH = '/path/to/metadata/metadata.csv'
_LABELS = list(np.genfromtxt('/path/to/labels/classes.txt', dtype='str'))
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class TestDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""
    # Version of the dataset, also name of the folder where you can find your dataset after building it
    VERSION = tfds.core.Version('1.0.0')

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata.
            Note about Audio: needs to have a defined shape (so no shape=(None, None) allowed)
            If your audio is ready to go you can decode it directly with :
            > 'audio': tfds.features.Audio(file_format='mp3', shape=(441000,2), dtype=tf.float32),
            If your audio data is all over the place like mine,
            I suggest you preprocess the audio like I've done in this example and pass it as a tensor:
        """
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'audio': tfds.features.Tensor(shape=(441000,), dtype=tf.float32),
                'label': tfds.features.ClassLabel(names=_LABELS),
                'artist': tfds.features.Text()
                }),
            # What goes in the model if as_supervised=True
            supervised_keys=('audio', 'label'),
            )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns Splits: do something that splits the data into at train and test"""
        df = pd.read_csv(_METADATAPATH)
        df_train = df.sample(frac=0.6, random_state=10) #random state is a seed value
        df_rest = df.drop(df_train.index).sample(frac=1.0)
        df_test = df_rest.sample(frac=0.5)
        df_val = df_rest.drop(df_test.index).sample(frac=1.0)

        return {
            'train': self._generate_examples(
                    (_AUDIODIR+df_train.filepath).values,
                    df_train.label.values,
                    df_train.artist_name.values,
                    df_train.filename.values,
                    ),
            'test': self._generate_examples(
                    (_AUDIODIR+df_test.filepath).values,
                    df_test.label.values,
                    df_test.artist_name.values,
                    df_test.filename.values,
                    ),
            'val': self._generate_examples(
                    (_AUDIODIR+df_val.filepath).values,
                    df_val.label.values,
                    df_val.artist_name.values,
                    df_val.filename.values,
                    ),
                }

    def _generate_examples(self, audio_paths, labels, artists, filenames):
        """Yields examples."""
        for path,label,artist,name in zip(audio_paths,labels,artists, filenames):
            yield name, {
                # If your audio is ready just pass in the path > 'audio': path
                # If not yield preprocessed audio NOT as a tensor but as numpy array...
                'audio':TestDataset._preprocess_audio(path).numpy(),
                'label': label,
                'artist': artist
                }

    # Some simple functions to pre-process the audio so it ends up in one channel, sampled at a given rate,
    # and of a given duration (you could also very well use librosa for this)
    @staticmethod
    def _preprocess_audio(path):
      audio, rate = TestDataset._load_audio(path)
      audio = TestDataset._tomono(audio)
      audio = TestDataset._resample(audio, rate_in=tf.cast(rate,tf.int64), rate_out=44100)
      return audio[:441000]

    @staticmethod
    def _load_audio(path):
        audio = tfio.IOTensor.graph(tf.float32).from_audio(path)
        return audio.to_tensor(), audio.rate

    @staticmethod
    def _tomono(audio_tensor):
        return tf.math.reduce_mean(audio_tensor, axis=1)

    @staticmethod
    def _resample(audio_tensor, rate_in, rate_out):
        return tfio.audio.resample(audio_tensor, rate_in=rate_in, rate_out=rate_out)
