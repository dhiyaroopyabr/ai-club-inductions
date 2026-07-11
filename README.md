# AI Club Induction Submissions - Dhiya Roopya B R

This repository contains the code, models, reports, and predictions submitted for the AI Club Induction process. It is organized into Round 2 (Memory-Efficient Exact Attention) and Round 3 (Typing Speed Forecasting).

---

## 📂 Repository Structure

The files in this repository are structured as follows:

*   **Round 2: Attention Mechanism Optimization**
    *   [`Round2_Code_DhiyaRoopyaBR.py`](./Round2_Code_DhiyaRoopyaBR.py): Python implementation of a memory-efficient exact attention mechanism.
    *   `Round2_Explanation_DhiyaRoopyaBR.pdf`: Detailed explanation of the chunked attention approach and backward pass strategy.
*   **Round 3: Typing Speed Forecasting**
    *   [`Round3_Model_DhiyaRoopyaBR.ipynb`](./Round3_Model_DhiyaRoopyaBR.ipynb): Jupyter notebook containing data exploration, model training, evaluation, and ensembling.
    *   [`Round3_Predictions_DhiyaRoopyaBR.csv`](./Round3_Predictions_DhiyaRoopyaBR.csv): The final forecasted predictions for typing speed metrics.
    *   `Round3_Report_DhiyaRoopyaBR.pdf`: A comprehensive report detailing data preprocessing, model selection, and performance metrics.

---

## 🚀 Round 2: Memory-Efficient Exact Attention

Standard attention mechanisms require storing an $O(n^2)$ attention matrix in memory, which quickly becomes a bottleneck for long sequences.

### Approach
1.  **Chunked Computation:** Processes query-key-value interactions in configurable chunks of size $C$ (`chunk_size`), reducing overall memory complexity to $O(n \times C)$.
2.  **Online Softmax:** Normalizes similarity scores on-the-fly using a running maximum and exponential sum, ensuring numerical stability without allocating the full attention grid.
3.  **Gradient Checkpointing:** Recomputes activations dynamically during the backward pass instead of saving intermediate matrices, drastically lowering training memory requirements at a negligible cost of recomputation.

### Running the Example
You can run the Python script to verify the memory-efficient forward pass with a sample input (matrix size $128 \times 32$):
```bash
python Round2_Code_DhiyaRoopyaBR.py
```

---

## 📈 Round 3: Typing Speed Forecasting

This task models and forecasts a user's typing metrics (Words Per Minute, Accuracy, Raw WPM, and Consistency) over time using historical training data.

### Hybrid Ensemble Model
To capture the unique temporal characteristics of each metric, a hybrid ensemble strategy was developed by training multiple models (ARIMA, Moving Averages, Last Value Baseline) and selecting the best performer per target feature:
*   **WPM (Words Per Minute):** Forecasted using a **Last Value Predictor** (captures the most recent performance baseline).
*   **Accuracy:** Forecasted using an **ARIMA(1,1,1)** model (effectively models trends and autoregressive error dependencies).
*   **Raw WPM:** Forecasted using a weighted combination of the **Last Value** and the **Median of the last 5 days**.
*   **Consistency:** Forecasted using the **Median of the last 5 days**.

### Outputs
*   The final forecasted values are saved to [`Round3_Predictions_DhiyaRoopyaBR.csv`](./Round3_Predictions_DhiyaRoopyaBR.csv).
