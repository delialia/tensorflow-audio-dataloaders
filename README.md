# Loading and Preparing Audio with Tensorflow 2.4
Different ways to load and prepare audio with Tensorflow that I understood and found neat. \
I've put this together to hopefully help out others using tf for audio and for my future self who will probably forget all of it... \
This is by no means a golden standard and if you know a different much better way of loading and processing audio with tf **please branch away and pull request!** thank you <3 \
\
**Bias Warning:** I mainly work with **music** audio files


## To the point - what can you find here?
- Two different ways to handle audio data:
  - tfio : "on-the-fly" audio loading and pre-processing with tensorflow
  - tfds : create a custom audio tf dataset, build it (TFRecords) and benefit from the tfds methods, e.g. simply load it using `tfds.load(yourdataset)`

- A tf pipeline following the awesome OCD pleasing template from [Deep Learning in Production](https://github.com/The-AI-Summer/Deep-Learning-In-Production) with a dummy model for audio classification to see the different techniques in action.

## How to use it?
This is meant as an example/guide/reminder not as a package. If you do however want to test some of it with your data, you need to modify the paths and parameters in:
- configs/config.py
- dataloaders/my_tdfs/my_tfds.py

## What I found : tfio and tfds

**tfio** \
\
\+ No waiting time as no "feature" pre computing required, just plug and train \
\+ Very flexible, good to try out different audio preprocessing approaches \
\
\- Memory intensive \
\- Longer epochs than tfds

**tfds** \
\
\+ Type checked -> less unwanted surprises while training \
\+ TFRecords -> quicker epochs, less memory required while training \
\
\- Need to specify the shape of your data before hand \
\- Initial computation of TFRecords -> not that flexible if you're not sure about your preprocessing \
\- Not that intuitive to use

## Questions to ask yourself before starting:
- Is my dataset supported by [tfds](https://www.tensorflow.org/datasets/catalog/overview) or [mirdata](https://github.com/mir-dataset-loaders/mirdata)?
- How big is my dataset? (is it worth the optimising effort? if so, is it so big that I might encounter memory issues? if so, is TFRecords my only option then?)
- Do I want to load audio "on-the-fly" (get flexibility for slower epochs) or ...
- pre-process the whole dataset before hand, trade flexibility and a big once-off waiting time for the flashing quick TFRecords?

## Folder Structure
- **configs** : `config.py` file with all the configuration info mainly for the tfio approach and for the model
- **dataloaders**:
  - `dataloader_tfio.py` : all the code to load and prep your data with tfio
  - `my_tfds` : you don't need to define a loader because tfds gives you one, but you do need to define and build your TFRecords dataset:
    - `my_tfds.py`: the code to define your custom dataset that you then need to build, for example by typing in your terminal `tfds build` \
    **warning:** The config paths for this are hardcoded in the script and not in config because it's just much easier to read in this case
    - `inspect_my_tfds.py`: some prints to check the custom tfds dataset
- **models**:
  - `base_model.py` : abstract class defining how a model should look like
  - `mlp_model.py` : inherits from the base_model and explicitly calls the loading for tfio or tfds given in as parameter; see that's the only difference between the two mains. Change the `build` method for a fancier model!
- **utils**:
  - `config.py` : json parser to read the configuration info in configs
- `main_tfds.py` : what you need to run to load the data using tfds and feed/train your model with it.  
- `main_tfio.py` : what you need to run to load the data using tfio and feed/train your model with it.  


## My motivation: how to make the most out of Tensorflow when handling audio?
You have your audio dataset, now you need to load it and do some pre-processing before feeding it to your fancy model. If your dataset is not supported by the amazing [mirdata](https://github.com/mir-dataset-loaders/mirdata) or tf you have to code up your data loader. \
The most straightforward way: to use python+librosa combo, run your data through that, save your input features in memory and pass them on to tf using something like:
`tf.data.Dataset.from_tensor_slices(you/readtogo/data/path)` \
This is all very good if your dataset is small and you are patient. Not my case; so I looked into different ways of letting tf do the data handling, a.k.a : use the optimised code produced by people much more skilled than myself.
