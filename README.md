# Battery-Timer
## A Foundation Model for Battery Discharge Capacity Degradation Forecasting

Battery-Timer is a domain-adapted time series foundation model tailored for lithium-ion battery capacity degradation prediction. Built upon the Transformer-based Timer architecture developed by Tsinghua University, Battery-Timer is fine-tuned using approximately 10GB of publicly available battery degradation datasets to encourage a degradation-aware representation learning. This enables the model to generalize well across varying battery chemistries, operational scenarios, and degradation profiles. Extensive validation on the in-house CycleLife-SJTUIE dataset demonstrates that Battery-Timer not only surpasses the performance of its untuned counterpart but also exhibits strong zero-shot learning (ZSL) capabilities across both constant current (CC) and constant current-constant voltage (CCCV) charging conditions. Despite its large parameter size, Battery-Timer serves as a powerful teacher model in the proposed knowledge distillation framework, significantly improving the generalization ability of lightweight expert models while preserving low computational costs.

## Dataset
### SJTUIE-cyclelife

### Open-source battery capacity degradation dataset for fine-tuning

```
@article{severson2019data,
  title={Data-driven prediction of battery cycle life before capacity degradation},
  author={Severson, Kristen A and Attia, Peter M and Jin, Norman and Perkins, Nicholas and Jiang, Benben and Yang, Zi and Chen, Michael H and Aykol, Muratahan and Herring, Patrick K and Fraggedakis, Dimitrios and others},
  journal={Nature Energy},
  volume={4},
  number={5},
  pages={383--391},
  year={2019},
  publisher={Nature Publishing Group UK London}
}

@misc{calce_battery_data,
  title        = {CALCE Battery Data Sets},
  author       = {Center for Advanced Life Cycle Engineering (CALCE)},
  year         = {2024},
  howpublished = {\url{https://calce.umd.edu/battery-data}},
  note         = {Accessed: 2025-04-29}
}

@article{preger2020degradation,
  title={Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions},
  author={Preger, Yuliya and Barkholtz, Heather M and Fresquez, Armando and Campbell, Daniel L and Juba, Benjamin W and Rom{\`a}n-Kustas, Jessica and Ferreira, Summer R and Chalamala, Babu},
  journal={Journal of The Electrochemical Society},
  volume={167},
  number={12},
  pages={120532},
  year={2020},
  publisher={IOP Publishing}
}

@article{wang2023large,
  title={Large-scale field data-based battery aging prediction driven by statistical features and machine learning},
  author={Wang, Qiushi and Wang, Zhenpo and Liu, Peng and Zhang, Lei and Sauer, Dirk Uwe and Li, Weihan},
  journal={Cell Reports Physical Science},
  volume={4},
  number={12},
  year={2023},
  publisher={Elsevier}
}
```
## Quick start

```python
quick_start_validation.ipynb
```
## Citation
If you find this repo helpful, please cite our paper.

## Contributors
If you have any questions or want to use the code, feel free to contact:
Chan.Joey (SJTU_Chan_Joey@outlook.com)
