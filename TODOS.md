# TODOS
- Rerun model with current best parameters
  -   default MS2 loss, predicting target
  -   default MS2 loss + MS1 loss, positive only constraint
- multichannel conditioning signal
- Look into the data to figure out why there are weird MS1s

1) ms2 mse loss predicting "noise"
2) ms2 mse loss predicting target directly
3) ms2 + ms1 mse loss using best version of 1/2?
4) Multichannel conditioning signal (repeat 1/2 with new signal?)

- 
- Optimize inference and data types with [torchao](https://pytorch.org/blog/pytorch-native-architecture-optimization/)
- Add eval metrics to WandB logging, separate from training
- continue from checkpoint - Saksham
- Add a raw mzML/tdf parser (timsrust_py03) - Josh
- Obtain another dataset for testing - Josh
  - Potentially use HeLa
  - Orbitrap from 2018
- Benchmark if time

# In Progress

# Completed
- ~~- Dataloaders for MS1+MS2 data~~
- ~~Implement base diffusion model from PyTorch~~
- ~~Adapter layer for MS data to input dimensions of above~~
- ~~Maybe custom training loop~~
- ~~Eval code~~
- ~~Update dataloader to perform grid split of data - Justin~~
- ~~Move sampling.py into codebase - Justin~~
- ~~Add learning rate scheduler (apopt from [AlphaPeptDeep](https://github.com/MannLabs/alphapeptdeep/blob/5cb3d2c8da526e38c6dd94f370409a751da282de/peptdeep/model/model_interface.py#L34-L162)) - Justin~~
- ~~Update Transformer model - Leon~~
- ~~Update/Test different loss functions - Leon~~
