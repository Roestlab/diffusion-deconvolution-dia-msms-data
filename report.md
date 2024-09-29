
# Problem Statement

In mass spectrometry  (data-independent acquisition), the fragmentation signal derived from precursor ions is convoluted due to the presence of multiple fragment ions from different precursor ions. This highly multiplexed signal is difficult to interpret and presents a challenge in peptide identification. We hypothesize that we can utilize diffusion to extract a clean MS2 peak map from a highly convoluted raw MS2 peak map, by conditioning on the MS1 peak map.

Below, is a demonstration of a highly convoluted raw MS2 peak map.

![](img/Screenshot%20from%202024-09-29%2011-43-34.png)

# Solution

## Modified Transformer with Diffusion

![](model_arch_2.png)

## Example

### Our Query MS1 Peak Map
![](img/Screenshot%20from%202024-09-29%2011-51-56.png)

### Extracted clean MS2 Peak Map
<div style="display: flex; justify-content: center;">
  <img src="img/Screenshot%20from%202024-09-29%2011-43-34.png" style="width: 45%; margin-right: 10px;">
  <img src="img/Screenshot%20from%202024-09-29%2011-53-33.png" style="width: 45%;">
</div>

### MS1 with Predicted MS2 Peak Map
<img width="367" alt="image" src="https://github.com/user-attachments/assets/b95b7015-68d2-4793-a684-3b02fb8a8380">
<img width="206" alt="image" src="https://github.com/user-attachments/assets/54e30cd1-59bf-47d2-8a00-254853c45910">



## Performance

![alt text](./img/train_los.png)

![alt text](./img/train_summary_tbl.png)

![alt text](./img/table_perf_sum.png)
