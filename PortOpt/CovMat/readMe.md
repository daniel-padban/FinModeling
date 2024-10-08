Project to optimize portfolio, maximizing return while minimizing risk.
The assets chosen for this porftolio are the assets in OMX Stockholm 30

Objectives & methods:
- Predict future returns using LSTM
- Adjust weights of covariance matrix using optimization algorithm

Cost function:
- Sharpe ratio of weighted assets

Design of model:
1. LSTM:
    - Input: Return of a previous period 
    - Hidden layer: Layers inside NN
    - Output layer: Predicted return.
    - Optimization using MSPE (Mean-percentage-square-error)
3. Optimization of portfolio weights using cost function and gradient descent
