"""Model config in json format"""
# >>>>>>>>>> CHANGE THIS WITH YOUR INFO <<<<<<<<<<<<<<<<<<
CFG = {
    "data": {
    # Regarding data loading ...
        # Where is the audio stored?
        "audio_dir": '/path/to/audio/mp3',
        # What is the folder pattern? i.e. where can I find the actual audio file?
        "folder_pattern": '*/*/*/*.mp3',
        # Where is the metadata linked to the audio?
        "metadata_path": '/path/to/metadata/metadata.csv',
        # Assuming there is a csv file that links the audio paths to the annotations,
        # What is the name of the column with the audio paths?
        "paths_col": 'filepath',
        # and the name of the column with the annotations?
        "label_col": 'label',
        # Where should I save the vocabulary of the annotations?
        "vocab_path": './',
        # In the case of loading all audio from the audio_dir,
        # should it be shuffled while creating a tensorflow dataset?
        "shuffle": False,
        # If using pre-defined splits:
        # What ratio of the dataset should I train on?
        'train_ratio': 0.6,
        # What ratio of the dataset should I use for validation?
        'val_ratio' : 0.1,
        # What ratio of the dataset should I use for testing?
        'test_ratio' : 0.3,
        # What is the desired audio sampling rate (target for resampling)?
        "sample_rate": 44100,
        # How many seconds of audio should the model consider?
        'duration_sec':10,
        # In the case of a time-frequency audio transformation
        # What should be the FFT size?
        'nfft':1024, #2048,
        # How many samples in every window (frame)?
        'win_size':1024, #2048,
        # How many samples should the window shift by?
        'hop_size':512, #256,
        # NOTE: %overlap between frames = 100*(win_size-hop_size)/win_size
    },
    "train": {
    # Regarding training...
        # How many samples should a batch contain?
        "batch_size": 64,
        # How many samples should I keep in the buffer for pre-fetching?
        "buffer_size": 1000,
        # How many times should I iterate over the data?
        "epoches": 1, #20,
        # How many different validation splits?
        "val_subsplits": 5,
        # Which optimiser should Keras use?
        "optimizer": {
            "type": "adam"
        },
        # What evalutation metric should Keras use?
        "metrics": ["accuracy"],
        # Where should I save the trained model?
        'model_path':'./',
    },
}
