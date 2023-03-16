# wandb-launch-demo
A demo repo for Weights &amp; Biases Launch examples

## Running

1. Train
- Train small model on small dataset to verify pipeline and convergence
- Train small model on full dataset
- Train large model on small dataset
- Train large model on full dataset

2. Eval
- Eval script to load a pretrained-model and run evaluation

## Launch Queue Config

```json
{
    "gpus": "all",
    "builder": {
        "base_image": "nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu18.04"
    }
}
```
To test: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04`
