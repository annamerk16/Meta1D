# Meta: Finding Dataset Bias using Language Models

## Meta 1D 
### 👥 Team Members
| Name             | GitHub Handle | Contribution                                                             |
|-------------------|----------------|--------------------------------------------------------------------------|
| Yuling Liu        | @yuuuu1ing     | Data exploration, visualization, overall project coordination            |
| Alisha Varma      | @alishav14     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Bhavana Chemuturi | @bhavanachem   | Data preprocessing, feature engineering, data validation                 |
| Anna Merkulova    | @annamerk16    | Model selection, hyperparameter tuning, model training and optimization  |
| Iid Maxamuud      | @abdifatah2023 | Model evaluation, performance analysis, results interpretation           |
| Daniela Suqui     | @danielasqui   | Model evaluation, performance analysis, results interpretation           |


## 🎯 Project Highlights

- Developed NLP techniques to detect and quantify social biases in text datasets using word embeddings and classification models.
- Identified biased semantic relationships and evaluated their impact on model predictions, supporting Meta’s Responsible AI initiatives.
- Created heatmaps, embedding plots, and bias trend metrics to clearly communicate findings.
- Delivered a fully documented, reproducible codebase aligned with ethical AI standards.


## 👩🏽‍💻 Setup and Installation

**Note:** This project was developed as a single comprehensive Jupyter notebook containing all code, analysis, and visualizations.

### How to Run
1. Clone the repository
2. Open `Meta1D_RedditBias.ipynb` 
3. Run all cells in sequence
4. All required installations and dependencies are included in the notebook


## 🏗️ Project Overview

This project was completed as part of the Break Through Tech AI Program in partnership with Meta.

### Objective

- Identify and analyze social biases present in NLP datasets, understand how they influence model outputs, and evaluate techniques for interpreting or mitigating bias.
  
### Scope

- Detect biased terms and associations
- Use word embeddings to visualize semantic relationships
- Train simple classifiers to observe bias-driven prediction patterns
- Document findings and provide visual tools for interpretability

### Real-World Significance

- Bias in training data can perpetuate harmful stereotypes and lead to inequitable AI systems. Because Meta's platforms serve billions globally, mitigating dataset bias is essential for safety, fairness, and compliance with ethical AI standards.


## 📊 Data Exploration

### Datasets Used

**RedditBias Dataset**
- **Source:** https://github.com/umanlp/RedditBias/tree/master
- **Size:** ~27,000 samples across five bias categories
- **Structure:** CSV files containing Reddit comments with bias annotations
  - comment: Text content from Reddit
  - bias_sent: Binary label (0 = not biased, 1 = biased)
  - bias_type: Category of bias (gender, orientation, race, religion)
- **Purpose:** Primary training dataset for bias detection models

**CrowS-Pairs Benchmark**
- **Source:** https://github.com/nyu-mll/crows-pairs
- **Size:** Filtered to relevant bias types (gender, race-color, religion, sexual orientation)
- **Structure:** Sentence pairs testing stereotypical vs. anti-stereotypical associations
  - sent_more: More stereotypical sentence
  - sent_less: Less stereotypical sentence
  - stereo_antistereo: Label indicating stereotypical nature
- **Purpose:** External validation dataset to test model generalization beyond Reddit-specific language 

### Data Preprocessing
**Text Cleaning**
- Removed empty or invalid entries
- Ensured all text fields were properly formatted as strings
- Retained only binary class labels, excluding any ambiguous annotations

**Dataset Splitting**
- Applied stratified sampling to maintain proportional representation of both bias labels and bias types
- Created custom stratification column combining bias_sent and bias_type for balanced splits
- Split: 80% training, 20% validation
- Random seed: 42 for reproducibility

**Tokenization**
- BERT tokenization: Used bert-base-uncased tokenizer with max_length=128, padding='max_length', truncation=True
- RoBERTa tokenization: Used roberta-base tokenizer with identical parameters
- Applied dynamic padding via DataCollatorWithPadding for efficient batching

**Pseudo-Labeling for Data Expansion**
- Separated labeled data from unlabeled data
- Implemented iterative pseudo-labeling with confidence threshold of 0.95
- Process:
  - Predicted labels on unlabeled samples using fine-tuned BERT model
  - Selected high-confidence predictions (probability ≥ 0.95)
  - Added pseudo-labeled samples to training set
  - Retrained model on expanded dataset
  - Repeated for 3 iterations maximum

**Class Imbalance Handling**
- Used stratified splitting to maintain natural class distribution across train/validation sets
- Verified proportional representation of all bias categories in both splits
- Applied class weights during hyperparameter tuning (weight_decay parameter)

**Key Assumptions**
- Binary classification (biased vs. not biased) is sufficient for detecting demographic bias
- Reddit comment language patterns generalize to other text sources (tested via CrowS-Pairs)
- High-confidence pseudo-labels (≥95% probability) are reliable for expanding training data
- Combining different bias types (gender, race, religion, orientation) into a single model improves overall bias detection capability


### Exploratory Data Analysis Insights

**Figure 1: Distribution of Bias Categories**
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/c6a682d5-f145-48c0-90d1-e86b71b9e5d7" />

*Key Insight: The RedditBias dataset shows significant class imbalance, with religion-related biases (religion2: ~10,500 samples) representing nearly 3x more data than gender (~3,000) or race (~3,000) biases.*  

--- 

**Figure 2: Distribution of Bias Labels**

<img width="549" height="393" alt="image" src="https://github.com/user-attachments/assets/b875afe0-5722-46d7-9cab-9ada5793721f" />

*Key Insight: The dataset shows class imbalance with 43% biased samples vs 57% non-biased samples, requiring stratified sampling to ensure representative train/validation splits.*


## 🧠 Model Development

### Model Selection Justification
We selected **BERT** and **RoBERTa** for the following reasons:
- Pre-trained transformer models with strong contextual understanding of language
- Proven effectiveness on text classification and bias detection tasks
- Different architectural approaches (BERT's masked language modeling vs. RoBERTa's dynamic masking) allow comparison of generalization capabilities
- Well-suited for detecting subtle linguistic biases in social media text
- Available through HuggingFace's transformers library with community support

### Technical Approach

**Architecture:**
- Fine-tuned BERT-base-uncased and RoBERTa-base models
- Added sequence classification head for binary bias detection (biased vs. not biased)
- Used AutoModelForSequenceClassification with 2 output labels
- No custom layer modifications: leveraged pre-trained architectures as-is

**Training Process:**
1. Initial fine-tuning on RedditBias dataset baseline hyperparameters
2. Hyperparameter optimization using Optuna framework (20 trials) to find optimal:
   - learning rate, batch size, number of epochs, weight decay, warmup steps
3. Applied pseudo-labeling iteration to expand training data:
   - Predicted labels on unlabeled Reddit comments with 95% confidence threshold
   - Added high-confidence predictions to training set
   - Retrained model on expanded dataset
4. External validation on CrowS-Pairs benchmark to assess generalization
5. Iterative refinement with early stopping and best model checkpointing

**Key Hyperparameters:**
BERT (Baseline):
- Learning rate: 2e-5
- Batch size: 16 (per device)
- Epochs: 3
- Weight decay: 0.01
- Max sequence length: 128
- Optimizer: AdamW
- Evaluation strategy: Per epoch

BERT (Tuned via Optuna):
- Learning rate: Optimized between 1e-5 and 5e-5
- Batch size: Optimized among 8, 16, 32
- Epochs: Optimized between 2-5
- Weight decay: Optimized between 0.0-0.1
- Warmup steps: Optimized between 0-1000

RoBERTa (Baseline):
- Learning rate: 1e-5 (lower than BERT to account for different architecture)
- Batch size: 16 (per device)
- Epochs: 3
- Weight decay: 0.05 (higher than BERT baseline)
- Max sequence length: 128

RoBERTa (Tuned via Optuna):
- Learning rate: Optimized between 1e-5 and 5e-5
- Batch size: Optimized among 8, 16, 32
- Epochs: Optimized between 4-7, higher range than BERT
- Weight decay: Optimized between 0.0-0.1
- Warmup steps: Optimized between 0-1000

## ⭐️ Code Highlights

- **Data Loading & Preparation (Cells 1-18):** Loading RedditBias datasets from multiple CSV files, combining bias categories, splitting into labeled/unlabeled data, stratified train/validation split
- **BERT Training Pipeline (Cells 30-54):** Tokenization with BERT tokenizer, model initialization, baseline training, hyperparameter tuning with Optuna, pseudo-labeling implementation, final model saving
- **CrowS-Pairs External Validation (Cells 57-62):** Loading and preprocessing CrowS-Pairs benchmark, evaluating BERT on external dataset, computing cross-dataset performance metrics
- **RoBERTa Training Pipeline (Cells 68-88):** RoBERTa tokenization, training on pseudo-labeled expanded dataset, hyperparameter optimization, model evaluation and comparison
- **RoBERTa External Validation (Cells 90-94):** Evaluating RoBERTa generalization on CrowS-Pairs, comparing cross-model performance
- **Comprehensive Visualizations (Cells 97-106):** Model performance comparisons, confusion matrices, bias type breakdowns, hyperparameter impact analysis, external dataset performance charts


## 📈 Results & Key Findings

### Model Performance Summary 

**RedditBias Dataset (Internal Validation)**

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **BERT (Tuned)** | **95.26%** | **95.88%** | **95.69%** | **96.06%** |
| **RoBERTa (Tuned)** | **82.58%** | **85.43%** | **81.92%** | **89.26%** |

*Performance on RedditBias validation set after hyperparameter optimization*


**CrowS-Pairs Dataset (External Validation)**

| Model | Overall Accuracy | F1-Score | Sample Count |
|-------|------------------|----------|--------------|
| **BERT** | **49.6%** | **61.2%** | 883 samples |
| **RoBERTa** | **49.84%** | **61.54%** | 967 samples |

*Cross-dataset generalization performance on external benchmark*

#### Performance by Bias Category (CrowS-Pairs)

| Bias Type | BERT Accuracy | RoBERTa Accuracy | BERT F1 | RoBERTa F1 | Better Model |
|-----------|---------------|------------------|---------|------------|--------------|
| **Gender** | 51.9% | 49.6% | 0.530 | 0.566 | BERT |
| **Race-Color** | 45.2% | 56.6% | 0.603 | 0.711 | RoBERTa (+11.4%) |
| **Religion** | 65.7% | 72.4% | 0.783 | 0.830 | RoBERTa (+6.7%) |
| **Sexual Orientation** | 52.4% | 59.5% | 0.649 | 0.721 | RoBERTa (+7.1%) |
| **AVERAGE** | **53.8%** | **59.5%** | **0.641** | **0.707** | **RoBERTa (+5.7%)** |

### Visual Results

**Figure 3: BERT vs RoBERTa Performance on RedditBias**

<img width="461" height="287" alt="Screenshot 2025-12-14 at 5 01 33 PM" src="https://github.com/user-attachments/assets/c8790149-8164-4d1b-9cb8-ecfc28b69768" />

*BERT significantly outperformed RoBERTa on the Reddit training dataset across all metrics, achieving 95.26% accuracy compared to RoBERTa's 82.58%.*

---

**Figure 4: Performance Breakdown by Bias Type**

<img width="482" height="282" alt="Screenshot 2025-12-14 at 5 02 12 PM" src="https://github.com/user-attachments/assets/5b47c97f-7cdb-4b46-b8a4-6452f595846f" />

*Performance varied significantly across demographic categories, with religion-based bias detection achieving highest accuracy (65.7-72.4%), correlating with larger training sample sizes (10,500+ samples).*

---

**Figure 5: Confusion Matrices - BERT vs RoBERTa**

<img width="635" height="275" alt="Screenshot 2025-12-14 at 5 03 30 PM" src="https://github.com/user-attachments/assets/0b5c395b-51c2-4581-afb8-374f0ba82791" />

**BERT (Tuned) Confusion Matrix Results:**
- True Negatives: 916 | True Positives: 1,219
- False Positives: 100 | False Negatives: 64

**RoBERTa (Tuned) Confusion Matrix Results:**
- True Negatives: 861 | True Positives: 1,157
- False Positives: 232 | False Negatives: 203

*BERT showed superior classification accuracy with fewer misclassifications, particularly achieving 64 false negatives compared to RoBERTa's 203.*

---

**Figure 6: Cross-Dataset Performance Comparison**

<img width="633" height="363" alt="Screenshot 2025-12-14 at 5 04 11 PM" src="https://github.com/user-attachments/assets/9000a8a0-2589-45e9-bd0f-5444ae49dd58" />


*Both models experienced significant performance degradation on external validation:*
- BERT: F1 dropped from 0.830 (Reddit) to 0.552 (CrowS-Pairs) - **28% decrease**
- RoBERTa: F1 dropped from 0.823 (Reddit) to 0.552 (CrowS-Pairs) - **27% decrease**

*This domain shift indicates models learned Reddit-specific linguistic patterns rather than fully generalizable bias detection capabilities.*

---

### Key Findings

**1. BERT Dominated on Reddit Data, But RoBERTa Generalized Better**
- **On RedditBias:** BERT achieved 95.26% accuracy vs RoBERTa's 82.58% - BERT won by **12.68 percentage points**
- **On CrowS-Pairs:** RoBERTa averaged 59.5% accuracy vs BERT's 53.8% - RoBERTa won by **5.7 percentage points**
- **Insight:** This represents a **generalization vs. memorization trade-off** - BERT memorized Reddit-specific patterns better, while RoBERTa learned more transferable bias concepts

**2. Cross-Dataset Generalization Remains a Major Challenge**
- Both models experienced ~33% F1-score drops when tested on CrowS-Pairs external dataset
- Performance on external validation (49-50% accuracy) barely exceeded random baseline (50%)
- Indicates models learned Reddit-specific linguistic patterns (slang, discourse style) rather than universal bias markers
- Domain shift between casual Reddit comments and constructed CrowS-Pairs sentences significantly impacted performance

**3. Training Data Size Directly Correlates with Detection Accuracy**
- **Religion biases** (10,500+ training samples) achieved highest accuracy: 65.7% (BERT), 72.4% (RoBERTa)
- **Sexual orientation** (8,000 samples) showed moderate performance: 52.4% (BERT), 59.5% (RoBERTa)
- **Gender & race** (~3,000 samples each) showed lowest performance: 45.2-51.9%
- **Clear pattern:** More training data = better bias detection capability

**4. Pseudo-Labeling Successfully Expanded Training Data**
- Completed 3 iterations with 95% confidence threshold
- **Iteration 1:** Added 5,673 high-confidence samples
- **Iteration 2:** Added 6,226 high-confidence samples  
- **Iteration 3:** Added 2,916 high-confidence samples
- **Total expansion:** ~15,000 additional training samples improved model robustness

**5. RoBERTa's Advantage on Underrepresented Categories**
- RoBERTa showed **+11.4% improvement on race-color** bias detection compared to BERT
- Dynamic masking and larger pre-training corpus helped generalize to racial bias patterns
- Demonstrates RoBERTa's strength in **few-shot learning** scenarios with limited training data

**6. Class Imbalance Significantly Impacts Model Performance**
- Dataset composition: ~60% biased vs ~40% not biased samples
- Religion2 category alone comprised nearly 40% of all training data
- Despite stratified sampling, models showed bias toward over-represented categories
- Performance disparity across categories highlights critical need for balanced datasets

## 💭 Discussion and Reflection

### What Worked
- **BERT's architecture proved highly effective** for capturing bias patterns in Reddit data, achieving 96.04% accuracy on the validation set - significantly outperforming our initial expectations and baseline models.
- **Pseudo-labeling successfully expanded training data** through iterative semi-supervised learning. Over 3 iterations with a 95% confidence threshold, we added high-confidence predictions from unlabeled Reddit comments, which improved model robustness and reduced overfitting to the initial labeled dataset.
- **Hyperparameter optimization using Optuna** yielded meaningful performance gains. Systematic tuning of learning rate, batch size, epochs, weight decay, and warmup steps improved F1-scores and helped both models converge more effectively.
- **Stratified sampling** maintained balanced representation across both bias labels and bias types (gender, race, religion, orientation), ensuring models learned to detect all forms of demographic bias rather than just the most common categories.
- **Multi-dataset validation** provided critical insights into model generalization beyond Reddit-specific language, revealing important limitations in cross-domain transfer.

### Challenges Encountered
- **RoBERTa's lower generalization:** Despite RoBERTa's theoretical advantages, it achieved only 89.05% accuracy compared to BERT's 96.04%. The ~7% performance gap suggests BERT's architecture may be better suited for the linguistic patterns in social media bias detection, or that RoBERTa requires different fine-tuning strategies.
- **Dataset specificity and domain shift:** Both models showed performance degradation on external validation compared to in-domain data, indicating potential overfitting to Reddit-specific language patterns. This highlights the challenge of building bias detectors that generalize across different text sources.
- **Class imbalance across bias categories:** Religious bias had 10,500+ samples while gender and race had only ~3,000 each. Despite stratified sampling, this imbalance may have caused models to be more sensitive to religious bias patterns while underperforming on underrepresented categories

### Why These Results Matter

- **For Meta's Responsible AI initiatives:** These findings demonstrate that while transformer models can effectively detect demographic bias in text, model selection and training strategies significantly impact real-world performance. 

- **For the broader NLP community:** The accuracy gap between BERT and RoBERTa highlights that newer or theoretically superior models don't always outperform on specialized tasks like bias detection. Task-specific evaluation is essential, and architectural choices should be validated empirically rather than assumed based on general benchmarks.

- **Moving forward:** These results emphasize the importance of external validation, diverse datasets, and careful consideration of how training data characteristics influence model behavior in bias-critical applications.

## 🚀 Next Steps

- Expand datasets (especially religion-related biases; add non-Reddit sources).
- Test larger transformer models (e.g., RoBERTa-large).
- Apply domain adaptation techniques to improve cross-dataset robustness.
- Conduct deeper error analysis to reduce false positives/negatives.
- Improve user-facing prototype for wider deployment.

## 📝 License
This project is licensed under the MIT License – see the LICENSE file for details.
## 🙏 Acknowledgements

- Megan Ung, Meta Challenge Advisor
- Candace Ross, Meta Challenge Advisor
- Rajshri Jain, AI Studio Coach
- Break Through Tech AI Program
- Peer collaborators and mentors who supported our project
