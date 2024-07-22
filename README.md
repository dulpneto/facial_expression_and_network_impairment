# Real-Time Multimedia Communication Assessment via Facial Expressions

## Overview

This repository contains the code and data associated with the paper "A Recurrent Neural Network Approach to Real-Time Multimedia Communication Assessment via Facial Expressions". The repository includes the dataset used in the study, as well as Jupyter notebooks for data processing, model training, and evaluation.

## Repository Structure

```
.
├── data
│   ├── raw
│   │   ├── user_1
│   │   │   ├── TC1_0-0-0.csv
│   │   │   ├── ...
│   │   └── ...
│   ├── processed
│   │   ├── X_train.npy
│   │   ├── Y_train.npy
│   │   ├── X_test.npy
│   │   └── Y_test.npy
│   └── README.md
├── notebooks
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_model_training_experiment_parameters.ipynb
│   ├── 3_model_training_webrtc_internals.ipynb
│   └── 4_model_evaluation.ipynb
└── README.md
```

## Getting Started

[//]: # (### Prerequisites)

[//]: # ()
[//]: # (To run the code in this repository, you need to have Python 3.7+ installed. You can install the required Python packages using the following command:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

### Data

The `data` directory contains the raw and processed data used in the study. The raw data is organized by user and test case, while the processed data is stored in NumPy arrays for easy loading during model training.

### Notebooks

The `notebooks` directory contains Jupyter notebooks for data preprocessing, model training, and evaluation:

1. **1_data_preprocessing.ipynb**: This notebook contains the code for preprocessing the raw data, including filtering, standardization, and sequence generation.
2. **2_model_training_experiment_parameters.ipynb**: This notebook contains the code for training the recurrent neural network model to predict experiment parameters (delay, jitter, and packet loss).
3. **3_model_training_webrtc_internals.ipynb**: This notebook contains the code for training the recurrent neural network model to predict WebRTC internals (video and audio packet loss and jitter).
4. **4_model_evaluation.ipynb**: This notebook contains the code for evaluating the trained models and visualizing the results.

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

To preprocess the raw data, run the `1_data_preprocessing.ipynb` notebook. This will generate the processed data files in the `data/processed` directory.

### Model Training

To train the models, run the `2_model_training_experiment_parameters.ipynb` and `3_model_training_webrtc_internals.ipynb` notebooks. These notebooks will save the trained models and training logs.

### Model Evaluation

To evaluate the trained models, run the `4_model_evaluation.ipynb` notebook. This will generate evaluation metrics and visualizations.

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
