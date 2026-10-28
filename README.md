
# Website Aesthetics Model: Multi-Task Learning for Website Aesthetics Assessment

## Overview

This repository contains a **multi-task learning model** designed to predict website aesthetics through two complementary tasks: rating prediction and pairwise comparison, on a CNN base pretrained for style recognition (Flickr Style).

*Code for my MEng thesis ([PDF, in Greek](https://publisync.issel.ee.auth.gr/assets/67b583412b6757a87d19c1ff/1752130907820-Germanopoulos_Thesis.pdf)), extending [Calista](https://github.com/calista-ai) (Delitzas et al., 2023).*

The system is particularly useful for content curation, website design evaluation, and automatic quality filtering in image-heavy applications.

## Tech Stack

- **Deep Learning Framework**: TensorFlow 2.15.0 & Keras 2.15.0
- **Computer Vision**: OpenCV 4.8.0, Pillow 10.3.0
- **Data Processing**: Pandas 2.2.2, NumPy 1.25.2, SciPy 1.13.1
- **ML Utilities**: Scikit-learn 1.4.2, Joblib 1.4.2
- **Visualization**: Matplotlib 3.9.0, Seaborn 0.13.2
- **Development**: Jupyter Notebook, IPython 8.18.1
- **Environment**: Python 3.9.19 (Conda)

## Features

### 1. **Multi-Task Learning Architecture**
   - **Shared Base**: Convolutional layers initialised from a network pretrained on Flickr Style
   - **Rating Head**: Predicts a 1–9 aesthetic score for a single website
   - **Comparison Head**: Siamese; the difference of the two scores passes through tanh to give the preferred website

### 2. **Transfer Learning**
   - Pretrained models from Flickr Style [dataset](https://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html) (217 MB) eliminate the need for training from scratch
   - Shared convolutional feature extraction across both tasks
   - Task-specific output heads for specialized predictions

### 3. **Comprehensive Dataset Support**
   - **Rating-based Dataset**: Direct aesthetic scores from user annotations
   - **Comparison-based Dataset**: Pairwise image comparisons with preference labels
   - **Data Validation**: Built-in checks for self-pairs and duplicate comparisons
   - Standardized CSV format for train/test splits

### 4. **Joint Training**
   - One model, two heads, trained together on a weighted sum of the two losses
   - MAE for rating, a Bradley-Terry logistic loss for comparison (weights 0.5 / 0.55)

## How This Project Was Built

### Phase 1: Data Collection & Preparation
- **Rating Task**: Used direct aesthetic ratings from user annotations, stored in the [website-aesthetics-datasets](https://github.com/Filipgerm/website-aesthetics-datasets) repository
- **Comparison Task**: Used a crowdsourcing application with active learning to collect a new comparison dataset on the same webpages. It can be found [here](https://github.com/Filipgerm/crowdsourcing-app)
- **Data Validation**: Implemented quality checks to ensure no self-pairs, duplicate comparisons, or train/test data leakage

### Phase 2: Model Architecture Design
- **Shared Base Network**: Utilized pretrained Flickr Style CNN as the feature extractor (convolutional layers)
- **Task-Specific Heads**: Designed two heads on the shared base:
  - Rating regression head (continuous value prediction)
  - Siamese comparison head (score difference → tanh, Bradley-Terry loss)
- **Multi-Task Training**: Combined both losses with fixed weights
- **Custom LRN Layer**: Wrote a Local Response Normalization layer for Keras, which has none built in

### Phase 3: Training & Validation
- **Transfer Learning**: Fine-tuned pretrained layers on the rating and comparison datasets
- **Cross-Task Learning**: Enabled feature sharing to improve generalization
- **Performance Comparison**: Analyzed individual task performance vs. the multi-task model (documented in `comparison.ipynb`)

### Phase 4: Joint Training
- Trained the multi-task model end to end (`Ensemble.ipynb`)
- Compared it with the single-task rating and comparison models

## Lessons Learned: Multi-Task Architecture Insights

### 1. **Synergistic Task Learning**
   - Shared representations between the rating and comparison tasks improved feature quality despite apparent task independence
   - The comparison task acted as a powerful regularizer by encouraging relative judgment consistency

### 2. **Complementary Prediction Signals**
   - **Style-pretrained features** capture high-level visual patterns (modern vs. classic design)
   - **Direct ratings** provide absolute aesthetic judgments
   - **Pairwise comparisons** learn fine-grained discrimination between similar images

### 3. **Transfer Learning Effectiveness**
   - Feature reuse from ImageNet → Flickr → target tasks created a powerful feature hierarchy
   - Fine-tuning the top convolutional layers improved task-specific performance without catastrophic forgetting

### 4. **Data Heterogeneity as a Strength**
   - Different annotation methodologies (absolute ratings vs. relative comparisons) captured distinct aspects of aesthetics

## How to Improve

### Short-term Improvements
1. **Architecture Optimization**
   - Explore attention mechanisms to learn task-specific feature weighting
   - Add batch normalization tuning for faster convergence

2. **Data Enhancement**
   - Expand comparison dataset with more diverse image categories
   - Add temporal consistency checks (same website over time)

3. **Evaluation Metrics**
   - Add inter-rater reliability analysis for collected ratings
   - Implement ranking metrics (NDCG, Spearman correlation) for comparison task
   - Create ablation studies quantifying each task's contribution

### Long-term Improvements
1. **Model Innovations**
   - Implement Vision Transformers instead of CNNs for modern architecture
   - Add explainability (attention maps, saliency analysis) to understand aesthetic features
   - Develop few-shot learning capabilities for new domains (e.g., mobile UI, 3D renders)

2. **Domain Extension**
   - Create interpretable feature importance scores for aesthetic elements
   - Build interactive visualization tools showing which image regions drive predictions

3. **Deployment Pipeline**
   - Create REST API for inference with batch processing support
   - Implement continuous model evaluation on new collected data
   - Add A/B testing framework to validate improvements before deployment

## How to Run

### Prerequisites
- Anaconda or Miniconda installed
- GPU recommended (NVIDIA CUDA 11.0+) for faster training
- ~50GB disk space for datasets and models

### Setup Environment
```bash
# Create conda environment from provided spec
conda env create -f environment.yml

# Activate environment
conda activate env
```

### Download Datasets
```bash
# Clone the rating-based dataset repository
git clone https://github.com/Filipgerm/website-aesthetics-datasets
cd website-aesthetics-datasets

# Note: You'll need to adjust paths in the notebooks to point to your dataset location
# Currently configured for relative paths - update as needed for your setup
```

### Run Notebooks
Rating Task (rating.ipynb)

```bash
jupyter notebook rating.ipynb
```
Trains the model on direct aesthetic rating predictions. View cell outputs to see training curves and evaluation metrics.

### Comparison Task (comparison.ipynb)

```bash
jupyter notebook comparison.ipynb
```
Trains the model on pairwise image comparison data. Includes performance comparison between individual tasks.

### Multi-Task Model (Ensemble.ipynb)

``` bash
jupyter notebook Ensemble.ipynb
```
Trains the shared base with the rating and comparison heads together. Includes evaluation and prediction visualization.

### Key Configuration Points
* **Data Paths:** Update the relative paths in each notebook to match your local directory structure (currently points to `../../Calista/website-aesthetics-datasets-master/`)
* **Model Weights:** Load pretrained Flickr Style weights from Caffe or convert from available TensorFlow checkpoints
* **Hyperparameters:** Modify batch size, learning rate, and task weights in the respective notebooks
* **Loss Weights:** Adjust the rating vs. comparison loss weights in `Ensemble.ipynb`

### Expected Results
* **Rating Task:** PCC 0.78 (95% CI 0.69–0.85), RMSE 0.626 on the held-out test set (1–9 scale)
* **Comparison Task:** 68.0% accuracy on 378 held-out preference pairs
* **Multi-Task Model:** PCC 0.786 (95% CI 0.70–0.85), RMSE 0.672, 69.8% comparison accuracy
* **Runtime:** ~5-10 minutes per notebook execution on GPU, 30-60 minutes on CPU

*The figures above are from the thesis (141-epoch run). The outputs saved in `Ensemble.ipynb` are from an exploratory 300-epoch run and show lower scores.*

### Troubleshooting
* **Memory Issues:** Reduce batch size in notebooks (search for batch_size = )
* **Missing Data:** Verify dataset paths match the cloned repository structure
* **Import Errors: Ensure all packages from environment.yml are properly installed by running conda env update -f environment.yml

### Related Repositories
[Website Aesthetics Datasets](https://github.com/Filipgerm/website-aesthetics-datasets) - Training and evaluation data
[Crowdsourcing App](https://github.com/Filipgerm/crowdsourcing-app) - Data collection interface for comparison judgments

### License
This project is provided as-is for research and educational purposes.



