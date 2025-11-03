# Comparing Computer Vision Models Through Their Interpretability

## Abstract
This research investigates a new method for comparing models’ interpretabilities quantitatively. Using the CIFAR-10 dataset, we conducted experiments by comparing the saliency maps of a standard and an adversarially robust model. For this comparison, we use SAM 2 and different interpretability methods such as Expected Gradients and XRAI.

Our findings provide valuable insights that help in better understanding the reasons behind the model’s decisions. Our future work will explore further the potential of pruning to improve the interpretability of adversarially robust models, with the aim of developing more transparent and reliable computer vision systems.

---

## Project Structure

| File / Folder               | Description                                                    |
|-----------------------------|----------------------------------------------------------------|
| `training/`                 | Folder containing training-related scripts                    |
| `training/standard_training.py` | Training pipeline for the baseline model                      |
| `README.md`                 | Project overview and documentation                             |
| `XRAI.py`                   | Implementation of the XRAI interpretability method             |
| `depgraph_pruning.ipynb`    | Experiments exploring pruning and model interpretability        |
| `exp_grads_differences.ipynb` | Comparison of Expected Gradients between models                |


---

## Methods Used
- SAM 2 – Segment Anything Model 2
- Expected Gradients – Gradient-based interpretability technique  
- XRAI – Region-based attribution method for visualizing important image regions  
- Pruning Techniques – Used to analyze effects on interpretability and robustness  

---

## Dataset
All experiments were conducted using the CIFAR-10 dataset, a standard benchmark for image classification and interpretability research.

---

## Future Work
- Explore pruning techniques to enhance interpretability in adversarially robust models  
- Apply the interpretability framework to larger architectures and datasets  
- Develop quantitative interpretability metrics for transparency and fairness  





