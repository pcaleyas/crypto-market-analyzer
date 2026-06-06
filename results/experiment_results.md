# Experiment Results & Considerations

This document details the approaches tested to solve the "predict zero" problem (collapse of the model due to trading frictions) and improve the LSTM sequence model.

## Tested Approaches

### 1. Baseline Model (Phase 3 - Random Forest)
- **Methodology**: Standard Random Forest Classifier trying to predict the binary `Realistic_Target` (1 if expected return > fees, 0 otherwise).
- **Results**: Average Sharpe Ratio of `-0.0240`, Average Max Drawdown of `-29.62%`.
- **Considerations**: The baseline model failed to overcome trading frictions. The complex, non-linear relationships in sequential market data are hard for simple tree splits on static days to capture.

### 2. Deep Learning Classification with Class Weights
- **Methodology**: LSTM network. We calculated `compute_class_weight='balanced'` and forced the neural network to penalize missing a profitable trade (False Negatives) more than usual.
- **Results**: Average Sharpe Ratio of `0.0478`, Average Max Drawdown of `-14.42%`.
- **Considerations**: A slight improvement, but it still suffered from the "predict zero" problem in Folds 3 and 4. Class weighting isn't enough when the market is extremely noisy; the model still prefers the safety of not trading.

### 3. Deep Learning Classification with Dynamic Threshold (The Best Performer)
- **Methodology**: LSTM network. Instead of a hard 0.5 probability cutoff to decide when to buy, we tested thresholds ranging from 0.30 to 0.70 on the **Validation Set** of each fold and selected the threshold that maximized the Sharpe Ratio.
- **Results**: Average Sharpe Ratio of `0.8965`, Average Max Drawdown of `-39.59%`.
- **Considerations**: This was highly successful. In Fold 4, adjusting the threshold to `0.30` resulted in a massive `274.90%` return. The neural network learns the pattern, but its confidence is naturally low (< 0.5). Dynamically lowering the barrier allows the network to execute on its real edge. 

### 4. Deep Learning Regression (The Initial Recommendation)
- **Methodology**: LSTM network with a `linear` output layer and `MSE` loss, tasked with predicting the exact `Expected_Return` of tomorrow. The trading rule is externalized: Buy if `predicted_return > FEE_RATE * 2 + SLIPPAGE`.
- **Results**: Average Sharpe Ratio of `-0.0071`, Average Max Drawdown of `-31.23%`.
- **Considerations**: While theoretically robust in finance, regression on daily Crypto returns is extremely difficult without extensive volatility scaling and normalization. The model struggled to predict the exact percentage move accurately enough, often under-predicting the magnitude of the move, which caused it to rarely cross our strict fee threshold. 

## Conclusion & Discarded Methods
We discard the **Regression** and **Class Weights** methods for this specific dataset. 
The **Classification with Dynamic Thresholding** proved to be by far the most profitable and adaptive approach. It teaches us a valuable lesson: LSTMs can find the signal, but their raw output probabilities are uncalibrated for trading. Optimizing the decision threshold on a validation set aligns the model's edge directly with the financial objective.
