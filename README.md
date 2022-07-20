# Stock-Market-Predictor
A CNN-LSTM based time series stock market predictor.  
[Original Research Paper](https://github.com/cyberbyte53/Stock-Market-Predictor/blob/main/CNN-LSTM.pdf)  
## Datasets
used two kinds of datasets 
1. High stock prices like Reliance(in the range of 1000)
2. Low stock prices like Transchem(in the range of 100)  
clean.sh => removes null data from dataset
## Models
stores saved models

## Results
stores plot of the data

## stock_market_predictor.ipynb
when want to analyse 1 stock market data
## stock_market_predictor.py
outputs results for all datasets and stores corresponding models

## Analysis
Based on plots from various datasets this method best forcasts stock prices when stock prices are relatively low.
For Example
![Very Accurate Prediction](https://github.com/cyberbyte53/Stock-Market-Predictor/blob/main/Results/TRANSCHEM.png)
![Bad Prediction](https://github.com/cyberbyte53/Stock-Market-Predictor/blob/main/Results/RELIANCE.png)
