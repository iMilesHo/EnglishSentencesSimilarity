## LSTM and Siamese based PitchTrainer similarity calculation

# Introduction
This project is a part of the PitchTrainer project. The goal of this project is to create a similarity calculation model for the PitchTrainer project. The model is based on the Siamese network architecture and uses LSTM layers to process the input data.

# Usage
## 1.1. Training and Testing
To train the model, run the following command:
```
python3 training.py
```
or just run the training.py file in your IDE.

## 1.2. Testing the model on a single custom example
To test the model, run the following command:
```
python3 main.py
```
or just run the main.py file in your IDE.

In this case, you can change the input data in the main.py file to your custom example.

# Architecture
## Similarity calculation model
```
Sub_SiameseNetwork_1(
  (embedding): Embedding(41708, 128)
  (lstm): LSTM(128, 128, batch_first=True)
)
Sub_SiameseNetwork_2(
  (embedding): Embedding(41708, 128)
  (lstm): LSTM(128, 128, batch_first=True)
)
```




