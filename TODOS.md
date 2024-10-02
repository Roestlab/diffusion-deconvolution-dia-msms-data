# TODOS
- Update dataloader to perform grid split of data
- Add learning rate scheduler (apopt from [AlphaPeptDeep](https://github.com/MannLabs/alphapeptdeep/blob/5cb3d2c8da526e38c6dd94f370409a751da282de/peptdeep/model/model_interface.py#L34-L162))
- Update/Test different loss functions
- Optimize inference and data types with [torchao](https://pytorch.org/blog/pytorch-native-architecture-optimization/)
- Add eval metrics to WandB logging
- Move sampling.py into codebase
- Add a raw mzML/tdf parser
- Obtain another dataset for testing
- Benchmark if time

# In Progress

# Completed
- ~~- Dataloaders for MS1+MS2 data~~
- ~~Implement base diffusion model from PyTorch~~
- ~~Adapter layer for MS data to input dimensions of above~~
- ~~Maybe custom training loop~~
- ~~Eval code~~