# APDM: Adaptive Physics-aware Diffusion Model for image dehazing
&emsp;Codes for paper “Adaptive Physics-Aware Diffusion Models for Patch-Based Image Dehazing”

## Pipeline
<center class ='img'>
<img title="Architecture of the proposed network." src="https://github.com/SYSUzrc/APDM/blob/main/insert/pipeline.png" width="90%">
</center>


## Evaluation
### Pre-trained Model Weights
&emsp;We have shared [a pre-trained model](https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar) on the NH-Haze dataset, with its configuration file designated as `nhaze.yml`. Please deposit it in the root directory of the project. <br /> 
&emsp;If you wish to test our model on alternative datasets, please retrain the model and organize the datasets in the following manner:
```
-- APDM-main
   -- dataset_name
      -- gt
      -- input
      -- dataset_name.txt
```

### Dataset Preparation
&emsp;We have prepared the [NH-haze](https://pan.baidu.com/s/1kjlPWNJuHgVXb8jk4nJ28w?pwd=1234)(Extraction Code : 1234), named in accordance with the code requirements. Please deposit it in the root directory of the project.
&emsp;
### Evaluating on the NH-haze Dataset

```bash
python eval_diffusion.py --config "nhaze.yml" --resume 'xxx.pth.tar' --sampling_timesteps 25 --grid_r 16
```

## NH-Haze Dataset
<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <th align="center">Input Condition</th>
    <th align="center">Restoration Process</th>
    <th align="center">Output</th>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/01.png" width="300" height="auto" alt="rd11"></td>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/02_demo.gif" width="300" height="auto" alt="rd12"></td>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/01_output.png" width="300" height="auto" alt="rd13"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/02.png" width="300" height="auto" alt="rd11"></td>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/01_demo.gif" width="300" height="auto" alt="rd12"></td>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/02_output.png" width="300" height="auto" alt="rd13"></td>
  </tr>
</table>

## Acknowledgmet
&emsp;Portions of this code repository are derived from the following works. Acknowledgement is extended to these seminal contributions:
* https://github.com/IGITUGraz/WeatherDiffusion
* https://github.com/ermongroup/ddim
