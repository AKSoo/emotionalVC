Implementation of [VAWGAN](https://github.com/JeremyCCHsu/vae-npvc/tree/vawgan) (conditional VAE-WGAN) for non-parallel voice conversion in PyTorch, re-purposed for emotional voice conversion on [SAVEE dataset](http://kahlan.eps.surrey.ac.uk/savee).

[VAWGAN Paper](https://arxiv.org/abs/1704.00849)

# Dependency
- Python 3.6 
  - tensorflow >= 1.5.0
  - PyTorch >= 0.4.0
  - PyWorld
  - librosa
  - soundfile

# Usage
```bash
# feature extraction
python analyzer.py \
--dir_to_wav dataset/savee/wav \
--dir_to_bin dataset/savee/bin

# collect stats
python build.py 
--train_file_pattern "dataset/savee/bin/*/*.bin" \
--corpus_name savee

# training
python train.py architecture-vawgan-savee.json savee

# conversion
python convert.py architecture-vawgan-savee.json savee THap
```

Adjust model and training parameters in architecture JSON file.

## Note
1. It can only do **1-to-1 VC**.
2. Setting `epoch_vawgan` in the architecture to 0 results in 1-to-1 VAE.
