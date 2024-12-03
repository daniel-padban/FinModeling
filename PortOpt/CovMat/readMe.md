## Portfolio optimization inspired by Modern Portfolio Theory.
### Overview
The contents of this directory contain the code used for my portfolio optimization project.
The file `AllocDemo.ipynb` contains a demo of the project with optimized weights and a graph of their performance.

The optimization algorithm uses three metrics as optimization goals:
- Calmar Ratio - To minimize sudden large losses
- Omega Ratio - To assess general performance of assets, gain/loss ratio
- Sortino Ratio - To minimize negative volatility
The parameters `alpha`, `beta` and `gamma` represent the weights for the calmar, omega and sortino ratios respectively.
The algorithm AdamW was used to optimize the asset weights, with a learning rate of 1e-3.


### How to use
1. Clone the repository
2. Install dependencies from `reqs.txt`
3. Adjust parameters for the optimization
4. Run all cells to optimze asset weights and get results for the desired period.
