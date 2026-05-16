
# Website Aesthetics Model: Multi-Task Learning Ensemble for Image Quality Assessment

## Overview

This repository contains a **multi-task learning ensemble model** designed to predict website aesthetics and image quality through three complementary learning paradigms: style recognition, rating prediction, and pairwise comparison. The model leverages transfer learning from the Flickr Style dataset and combines multiple neural network architectures to achieve robust aesthetic assessment across diverse web imagery.

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
   - **Style Recognition Task**: Leverages pretrained convolutional layers from the Flickr Style dataset for visual style classification
   - **Rating Prediction Task**: Direct aesthetic rating prediction from user-curated datasets
   - **Pairwise Comparison Task**: Learns aesthetic preferences through image pair comparisons (absolute vs. relative judgments)

### 2. **Transfer Learning**
   - Pretrained models from Flickr Style [dataset](https://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html) (217 MB) eliminate the need for training from scratch
   - Shared convolutional feature extraction across all three tasks
   - Task-specific output heads for specialized predictions

### 3. **Comprehensive Dataset Support**
   - **Rating-based Dataset**: Direct aesthetic scores from user annotations
   - **Comparison-based Dataset**: Pairwise image comparisons with preference labels
   - **Data Validation**: Built-in checks for self-pairs and duplicate comparisons
   - Standardized CSV format for train/test splits

### 4. **Ensemble Predictions**
   - Combines predictions from all three tasks for robust aesthetic assessment
   - Flexible weighting schemes for task combination
   - Cross-validation across different rating and comparison methodologies

## How This Project Was Built

### Phase 1: Data Collection & Preparation
- **Rating Task**: Gathered direct aesthetic ratings from user annotations stored in the [website-aesthetics-datasets](https://github.com/Filipgerm/website-aesthetics-datasets) repository
- **Comparison Task**: Built a crowdsourcing application to collect pairwise image comparisons, enabling relative aesthetic judgment collection. It can be found [here](https://github.com/Filipgerm/crowdsourcing-app)
- **Data Validation**: Implemented quality checks to ensure no self-pairs, duplicate comparisons, or train/test data leakage

### Phase 2: Model Architecture Design
- **Shared Base Network**: Utilized pretrained Flickr Style CNN as the feature extractor (convolutional layers)
- **Task-Specific Heads**: Designed three independent output layers:
  - Style classification head (multi-class softmax)
  - Rating regression head (continuous value prediction)
  - Comparison ranking head (triplet or pairwise loss)
- **Multi-Task Training**: Combined losses from all three tasks with learnable or fixed weights

### Phase 3: Training & Validation
- **Transfer Learning**: Fine-tuned pretrained layers on the rating and comparison datasets
- **Cross-Task Learning**: Enabled feature sharing to improve generalization
- **Performance Comparison**: Analyzed individual task performance vs. ensemble predictions (documented in `comparison.ipynb`)

### Phase 4: Ensemble Integration
- Combined predictions from all three trained models with weighted averaging
- Explored different ensemble strategies documented in `Ensemble.ipynb`
- Validated ensemble robustness across the full image dataset

## Lessons Learned: Ensemble Architecture Insights

### 1. **Synergistic Task Learning**
   - Multi-task learning provided regularization benefits; models trained on single tasks overfitted more than ensemble variants
   - Shared representations between style, rating, and comparison tasks improved feature quality despite apparent task independence
   - The comparison task acted as a powerful regularizer by encouraging relative judgment consistency

### 2. **Complementary Prediction Signals**
   - **Style recognition** captures high-level visual patterns (modern vs. classic design)
   - **Direct ratings** provide absolute aesthetic judgments
   - **Pairwise comparisons** learn fine-grained discrimination between similar images
   - Ensemble combination reduced individual task biases and improved calibration

### 3. **Transfer Learning Effectiveness**
   - Pretrained Flickr Style layers significantly accelerated convergence (3-5x faster training)
   - Feature reuse from ImageNet → Flickr → target tasks created a powerful feature hierarchy
   - Fine-tuning the top convolutional layers improved task-specific performance without catastrophic forgetting

### 4. **Data Heterogeneity as a Strength**
   - Different annotation methodologies (absolute ratings vs. relative comparisons) captured distinct aspects of aesthetics
   - Combining heterogeneous signals was more effective than collecting more data of a single type
   - Validation helped identify systematic annotator biases that ensemble averaging mitigated

### 5. **Computational Trade-offs**
   - Ensemble inference required running three models (3x computational cost)
   - Distillation or model pruning could reduce this, but accuracy gains from ensembling justified the cost for offline applications
   - Batch processing and GPU optimization became critical for large-scale deployment

## How to Improve

### Short-term Improvements
1. **Architecture Optimization**
   - Implement knowledge distillation to compress the ensemble into a single model
   - Explore attention mechanisms to learn task-specific feature weighting
   - Add batch normalization tuning for faster convergence

2. **Data Enhancement**
   - Expand comparison dataset with more diverse image categories
   - Implement active learning to focus data collection on uncertain predictions
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
   - Fine-tune separate ensemble models for specific domains (e-commerce, blogs, portfolios)
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

### Ensemble Model (Ensemble.ipynb)

``` bash
jupyter notebook Ensemble.ipynb
```
Combines predictions from all three tasks. Demonstrates ensemble inference and prediction visualization.

### Key Configuration Points
* **Data Paths:** Update the relative paths in each notebook to match your local directory structure (currently points to `../../Calista/website-aesthetics-datasets-master/`)
* **Model Weights:** Load pretrained Flickr Style weights from Caffe or convert from available TensorFlow checkpoints
* **Hyperparameters:** Modify batch size, learning rate, and task weights in the respective notebooks
* **Ensemble Weights:** Adjust the weighting scheme in `Ensemble.ipynb` to prioritize different tasks (e.g., favor comparison over style)

### Expected Results
* **Rating Task:** Mean Absolute Error ~0.5-0.8 on held-out test set (depending on rating scale)
* Comparison Task:** Accuracy 70-85% on preference prediction
* **Ensemble:** Improved robustness and reduced prediction variance across all images
* **Runtime:** ~5-10 minutes per notebook execution on GPU, 30-60 minutes on CPU

### Troubleshooting
* **Memory Issues:** Reduce batch size in notebooks (search for batch_size = )
* **Missing Data:** Verify dataset paths match the cloned repository structure
* **Import Errors: Ensure all packages from environment.yml are properly installed by running conda env update -f environment.yml

### Related Repositories
[Website Aesthetics Datasets](https://github.com/Filipgerm/website-aesthetics-datasets) - Training and evaluation data
[Crowdsourcing App](https://github.com/Filipgerm/crowdsourcing-app) - Data collection interface for comparison judgments

### License
This project is provided as-is for research and educational purposes.



