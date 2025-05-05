# Battery-Timer
## A Foundation Model for Battery Discharge Capacity Degradation Forecasting

Battery-Timer is a domain-adapted time series foundation model tailored for lithium-ion battery capacity degradation prediction. Built upon the Transformer-based Timer architecture developed by Tsinghua University, Battery-Timer is fine-tuned using approximately 10GB of publicly available battery degradation datasets to encourage a degradation-aware representation learning. This enables the model to generalize well across varying battery chemistries, operational scenarios, and degradation profiles. Extensive validation on the in-house CycleLife-SJTUIE dataset demonstrates that Battery-Timer not only surpasses the performance of its untuned counterpart but also exhibits strong zero-shot learning (ZSL) capabilities across both constant current (CC) and constant current-constant voltage (CCCV) charging conditions. Despite its large parameter size, Battery-Timer serves as a powerful teacher model in the proposed knowledge distillation framework, significantly improving the generalization ability of lightweight expert models while preserving low computational costs.

## Dataset
### SJTUIE-cyclelife

### Open-source battery capacity degradation dataset for fine-tuning

## Quick start

```python
quick_start_validation.ipynb
```
## Citation
If you find this repo helpful, please cite our paper.

## Contributors
If you have any questions or want to use the code, feel free to contact:
Chan.Joey (SJTU_Chan_Joey@outlook.com)
