# APDM: Adaptive Physics-aware Diffusion Model for image dehazing

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

&emsp;🎖️Codes for paper “**Adaptive Physics-Aware Diffusion Models for Patch-Based Image Dehazing**”  <br />


$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$  _Ruicheng Zhang, Kanghui Tian, Xiangkun Shi, Yuhan Huang, Luwei Tu, and Zhi Jin*_ <br />

## Pipeline
<center class ='img'>
<img title="Architecture of the proposed network." src="https://github.com/SYSUzrc/APDM/blob/main/insert/pipeline.png" width="100%">
</center>

## Demo
<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <th align="center">Input Condition</th>
    <th align="center">Restoration Process</th>
    <th align="center">Output</th>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/01.png" width="310" height="auto" alt="rd11"></td>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/02_demo.gif" width="310" height="auto" alt="rd12"></td>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/01_output.png" width="310" height="auto" alt="rd13"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/02.png" width="310" height="auto" alt="rd11"></td>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/01_demo.gif" width="310" height="auto" alt="rd12"></td>
    <td align="center"><img src="https://github.com/SYSUzrc/APDM/blob/main/insert/02_output.png" width="310" height="auto" alt="rd13"></td>
  </tr>
</table>


## Installation
### Create a Virtual Environment
```
conda create --name APDM python=3.7.12
conda activate APDM
```

### Install Packages
```
pip install -r requirements.txt
```

## Evaluation
### Pre-trained Model Weights
&emsp;We have shared [a pre-trained model](https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar) on the NH-Haze dataset, with its configuration file designated as `nhaze.yml`. Please place it in the `./ckpts` folder. <br /> 
&emsp;If you wish to test our model on alternative datasets, please retrain the model and organize the datasets in the following manner:
```
|-- APDM-main
|  |-- dataset_name
|  |  |-- gt
|  |  |-- input
|  |  |-- dataset_name.txt
```

### Dataset Preparation
&emsp;We have prepared the [NH-haze](https://pan.baidu.com/s/1kjlPWNJuHgVXb8jk4nJ28w?pwd=1234)(Extraction Code : 1234), named in accordance with the code requirements. Please deposit it in the root directory of the project.
&emsp;
### Evaluating on the NH-haze Dataset

```bash
python eval_diffusion.py --config "test.yml" --resume 'Nhaze_ddpm.pth.tar' --sampling_timesteps 25 --grid_r 4
```


## Author
📧 : zhangrch23@mail2.sysu.edu.cn

## Acknowledgment
&emsp;This work was supported by the National Natural Science Foundation of China under Grant No.62071500; Supported by Supported by Shenzhen Science and Technology Program under Grant No. JCYJ20230807111107015. Supported by Fundamental Research Funds for the Central Universities, Sun Yat-sen University under Grant No. 241gqb015.
&emsp;*Zhi Jin is the corresponding author. <br />
&emsp;Portions of this code repository are derived from the following works. Acknowledgement is extended to these seminal contributions:
* https://github.com/IGITUGraz/WeatherDiffusion
* https://github.com/ermongroup/ddim


<!-- links -->
[your-project-path]: https://github.com/SYSUzrc/APDM
[contributors-shield]: https://img.shields.io/github/contributors/SYSUzrc/APDM.svg?style=flat-square 
[contributors-url]: https://github.com/SYSUzrc/APDM/graphs/contributors 
[forks-shield]: https://img.shields.io/github/forks/SYSUzrc/APDM.svg?style=flat-square 
[forks-url]: https://github.com/SYSUzrc/APDM/network/members 
[stars-shield]: https://img.shields.io/github/stars/SYSUzrc/APDM.svg?style=flat-square 
[stars-url]: https://github.com/SYSUzrc/APDM/stargazers 
[issues-shield]: https://img.shields.io/github/issues/SYSUzrc/APDM.svg?style=flat-square 
[issues-url]: https://github.com/SYSUzrc/APDM/issues 
[license-shield]: https://img.shields.io/github/license/SYSUzrc/APDM.svg?style=flat-square 
[license-url]: https://github.com/SYSUzrc/APDM/blob/main/LICENSE.txt 
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555 
[linkedin-url]: https://linkedin.com/in/zhangruicheng 
