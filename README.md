# deepspeech

Implementation of DeepSpeech2 using [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc).
Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture, trained with the CTC activation function.

## Features

* Train DeepSpeech, configurable RNN types and architectures with multi-gpu support.
* Language model support using kenlm (WIP right now, currently no instructions to build a LM yet).
* Multiple dataset downloaders, support for AN4, TED, Voxforge and Librispeech. Datasets can be merged, support for custom datasets included.
* Noise injection for online training to improve noise robustness.
* Audio augmentation to improve noise robustness.
* Easy start/stop capabilities in the event of crash or hard stop during training.
* Visdom/Tensorboard support for visualizing training graphs.

# Installation

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu.

Install PyTorch 0.4.0

Install this fork for Warp-CTC bindings:
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
export CUDA_HOME="/XXX/"
cd ../pytorch_binding
python setup.py install
```

Install pytorch audio:
```
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio
pip install cffi
python setup.py install
```

If you want decoding to support beam search with an optional language model, install ctcdecode:
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

Finally clone this repo and run this within the repo:
```
pip install -r requirements.txt
```


# Usage

## Dataset

Currently supports AN4, TEDLIUM, Voxforge and LibriSpeech. Scripts will setup the dataset and create manifest files used in dataloading.

### AN4

To download and setup the an4 dataset run below command in the root folder of the repo:

```
cd data; python an4.py
```


### LibriSpeech

To download and setup the LibriSpeech dataset run the below command in the root folder of the repo:

```
cd data; python librispeech.py
```



```
cd data/
python librispeech.py --files-to-use "train-clean-100.tar.gz, train-clean-360.tar.gz,train-other-500.tar.gz, dev-clean.tar.gz,dev-other.tar.gz, test-clean.tar.gz,test-other.tar.gz"
```


## Training

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv

FOR an4
python train.py  --rnn-type lstm --hidden-size 800 --hidden-layers 5 --checkpoint --train-manifest data/an4_train_manifest.csv --val-manifest data/an4_val_manifest.csv --epochs 100 --num-workers 2 --cuda --batch-size 20 --learning-anneal 1.01 --augment
```

Use `python train.py --help` for more parameters and options.

There is also [Visdom](https://github.com/facebookresearch/visdom) support to visualize training. Once a server has been started, to use:

```
python train.py --visdom
```

There is also [Tensorboard](https://github.com/lanpa/tensorboard-pytorch) support to visualize training. Follow the instructions to set up. To use:

```
python train.py --tensorboard --logdir log_dir/ # Make sure the Tensorboard instance is made pointing to this log directory
```

For both visualisation tools, you can add your own name to the run by changing the `--id` parameter when training.



### Checkpoints

Training supports saving checkpoints of the model to continue training from should an error occur or early termination. To enable epoch
checkpoints use:

```
python train.py --checkpoint
```

To enable checkpoints every N batches through the epoch as well as epoch saving:

```
python train.py --checkpoint --checkpoint-per-batch N # N is the number of batches to wait till saving a checkpoint at this batch.
```

Note for the batch checkpointing system to work, you cannot change the batch size when loading a checkpointed model from it's original training
run.

To continue from a checkpointed model that has been saved:

```
python train.py --continue-from models/deepspeech_checkpoint_epoch_N_iter_N.pth
```

This continues from the same training state as well as recreates the visdom graph to continue from if enabled.

If you would like to start from a previous checkpoint model but not continue training, add the `--finetune` flag to restart training
from the `--continue-from` weights.

### Choosing batch sizes

Included is a script that can be used to benchmark whether training can occur on your hardware, and the limits on the size of the model/batch
sizes you can use. To use:

```
python benchmark.py --batch-size 32
```

Use the flag `--help` to see other parameters that can be used with the script.

### Model details

Saved models contain the metadata of their training process. To see the metadata run the below command:

```
python model.py --model-path models/deepspeech.pth
```

To also note, there is no final softmax layer on the model as when trained, warp-ctc does this softmax internally. This will have to also be implemented in complex decoders if anything is built on top of the model, so take this into consideration!

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model-path models/deepspeech.pth --test-manifest /path/to/test_manifest.csv --cuda
```

An example script to output a transcription has been provided:

```
python transcribe.py --model-path models/deepspeech.pth --audio-path /path/to/audio.wav
```

## Server

Included is a basic server script that will allow post request to be sent to the server to transcribe files.

```
python server.py --host 0.0.0.0 --port 8000 # Run on one window

curl -X POST http://0.0.0.0:8000/transcribe -H "Content-type: multipart/form-data" -F "file=@/path/to/input.wav"
```

### Alternate Decoders
By default, `test.py` and `transcribe.py` use a `GreedyDecoder` which picks the highest-likelihood output label at each timestep. Repeated and blank symbols are then filtered to give the final output.

A beam search decoder can optionally be used with the installation of the `ctcdecode` library as described in the Installation section. The `test` and `transcribe` scripts have a `--decoder` argument. To use the beam decoder, add `--decoder beam`. The beam decoder enables additional decoding parameters:
- **beam_width** how many beams to consider at each timestep
- **lm_path** optional binary KenLM language model to use for decoding
- **alpha** weight for language model
- **beta** bonus weight for words

### Time offsets

Use the `--offsets` flag to get positional information of each character in the transcription when using `transcribe.py` script. The offsets are based on the size
of the output tensor, which you need to convert into a format required.
For example, based on default parameters you could multiply the offsets by a scalar (duration of file in seconds / size of output) to get the offsets in seconds.

