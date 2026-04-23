### Data & Code Availability

- **Metal Price Datasets**: 
`datasets/{metal}/{metal}_prices.csv`

- **Prediction Generation**  
  Notebook: `analysis/01_run_predictions.ipynb`  
  Output: Base model predictions stored in  
  `analysis/{metal}_shift_20d/predictions/test/`

- **Rule Filtering & EDCR Processing**  
  Notebook: `analysis/edcr/03_edcr_rule_filtering.ipynb`  
  Output: Processed predictions and evaluation results stored in  
  `analysis/edcr/evaluation_results/top_f1/` and  
  `analysis/edcr/evaluation_results/threshold/`

- **Ablation Study (Figure 1)**  
  Notebook: `analysis/edcr/04_ablation.ipynb`  

- **Model Evaluation Results (Table 2 & 3)**  
  Notebook: `analysis/edcr/05_model_evaluation_latex.ipynb`  