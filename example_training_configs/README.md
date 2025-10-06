In this directory, we provide some example scripts to set up environments on a Compute Canada cluster, and to run experiments.

## Network Weights & Trainings

Some of the experiments are conducted by fine-tuning models trained on the WQI dataset.  During the training phase, it is assumed both the checkpoint for the degradation model ($e_l, e_g$ and $\hat{f}$ in our paper) and that for the discriminator are presented.  However, due to the system update on Beluga cluster, we lost all intermediate checkpoints.  Hence, we are only able to provide the final checkpoints for the degradation model.

If you want to fine-tune our model, it is advisable to freeze the degradation model and only train the discriminator for a few epochs before the main training phase. 

## Environment Setup

The `env_setup.sh` is provided for reference only.  This one is tested to be working on Rorqual, but we are not sure whether it will produce the same result as on Beluga.  The original environment setup script is `env_setup_beluga.sh`. 


## Example for Training Config

An example training config is provided in `film_grain.yaml`. 


