# Real-Time Multimedia Communication Assessment via Facial Expressions


## Overview

This repository contains the code and data associated with the paper "A Recurrent Neural Network Approach to Real-Time Multimedia Communication Assessment via Facial Expressions", submitted to IEEE Latincom 2024. The repository includes the dataset used in the study, as well as Jupyter notebooks for data processing, model training, and evaluation.

## Paper abstract

The rapid adoption of video calls for remote work and virtual collaboration has highlighted the critical need for high-quality user experiences in real-time communication platforms. Traditional methods for assessing users' Quality of Experience (QoE) often rely on subjective user feedback, which can be inconsistent and difficult to quantify. We propose a novel approach that leverages facial expression analysis to predict network instabilities during video calls. Our methodology integrates facial expression data with network performance metrics to predict potential instabilities in real-time.

Using facial expression data collected during video calls, we employed a multi-output recurrent neural network model to predict network impairments. Our results demonstrate that the model effectively identifies facial expression features associated with network impairments, achieving high accuracy rates, particularly in predicting video-related issues (for instance, 91.30% accuracy for video packet loss). This indicates the potential of facial expression analysis as a reliable predictor of network performance issues, offering a more objective and user-centric assessment of QoE. In addition to the model, we present an Exploratory Data Analysis that discusses some precautions before feeding a learning model with a dataset. To facilitate the reproduction of our proposal, we used an openly available dataset, and all the code developed by us is shared as open source software.



## Repository Structure

```
.
├── data
│   ├── raw
│   │   ├── Facial_expression_features_dataset
│   │   │   ├── User1
│   │   │   │   ├── TC1_0-0-0.csv
│   │   │   │   ├── ...
│   │   │   └── ...
├── notebooks
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_model_training_experiment_parameters.ipynb
│   └── 3_model_training_webrtc_internals.ipynb
└── README.md
```


### Data

The `data` directory contains the raw and processed data used in the study. The raw data is organized by user and test case, while the processed data is stored in NumPy arrays for easy loading during model training.

### Notebooks

The `notebooks` directory contains Jupyter notebooks for data preprocessing, model training, and evaluation:

1. **1_data_preprocessing.ipynb**: This notebook contains the code for preprocessing the raw data, including filtering, standardization, and sequence generation.
2. **2_model_training_experiment_parameters.ipynb**: This notebook contains the code for training the recurrent neural network model to predict experiment parameters (delay, jitter, and packet loss).
3. **3_model_training_webrtc_internals.ipynb**: This notebook contains the code for training the recurrent neural network model to predict WebRTC internals (video and audio packet loss and jitter).

[//]: # (### Source Code)

[//]: # ()
[//]: # (The `src` directory contains the source code for data preprocessing, model training, and evaluation:)

[//]: # ()
[//]: # (- **data_preprocessing.py**: Functions for preprocessing the raw data.)

[//]: # (- **model_training.py**: Functions for training the recurrent neural network models.)

[//]: # (- **model_evaluation.py**: Functions for evaluating the trained models.)

[//]: # (- **utils.py**: Utility functions used throughout the project.)

## Usage

### Data Preprocessing

To preprocess the raw data, run the `1_data_preprocessing.ipynb` notebook. 

### Model Training

To train the models, run the `2_model_training_experiment_parameters.ipynb` and `3_model_training_webrtc_internals.ipynb` notebooks. These notebooks will save the trained models and training logs.

## Acknowledgments

This research is part of the INCT of the Future Internet for Smart Cities funded by CNPq proc. 465446/2014-0, Coordenação de Aperfeiçoamento de Pessoal de Nível Superior – Brasil (CAPES) – Finance Code 001, FAPESP proc. 14/50937-1, and FAPESP proc. 15/24485-9. It is also part of the FAPESP proc. 21/06995-0.

## Citation

If you use this code or data in your research, please cite the following paper:

```
@article{Neto2023RealTime,
  title={A Recurrent Neural Network Approach to Real-Time Multimedia Communication Assessment via Facial Expressions},
  author={Eduardo Lopes Pereira Neto, Sergio Hayashi, Daniel Macedo Batista, R. Hirata Jr., Nina S. T. Hirata, Karina Valdivia Delgado},
  journal={arXiv preprint arXiv:(not defined yet},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact Eduardo Lopes Pereira Neto at dulpneto@ime.usp.br.
