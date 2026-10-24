# Local Outlier Factor (LOF) Implementation
Custom Python implementation of the Local Outlier Factor (LOF) algorithm from scratch for MTH 4330. Includes data preprocessing, hyperparameter tuning, and baseline evaluation.

## Project Scope
* **Core Algorithm:** Local Outlier Factor (LOF) built from scratch in Python/NumPy.
* **Baselines:** k-Nearest Neighbors (k-NN) distance, PCA reconstruction error, and Isolation Forest.
* **Datasets:** ODDS Annthyroid, ADBench Tabular Suite (cardio), and Kaggle Credit Card Fraud.
* **Evaluation Metrics:** Average Precision, ROC-AUC, and Precision@n.

## Repository Structure
* `/data/`: Contains raw and processed data splits (ignored by git due to file size).
* `/src/`: Core Python modules for the LOF model, data pipelines, tuning, and evaluation.
* `/baselines/`: Scripts for running comparison models.
* `/tests/`: Unit tests and synthetic edge-case examples.

## Environment Setup
To ensure consistency across local machines and GitHub Codespaces, follow these steps to set up the development environment.

1. **Clone the repository:**

   git clone https://github.com/RostyslavKharevych/mth4330-lof-implementation.git
   cd mth4330-lof-implementation

3. Create a virtual environment:
* Windows: python -m venv venv
* macOS/Linux/Codespaces: python3 -m venv venv

3. Activate the virtual environment:
* Windows: .\venv\Scripts\activate
* macOS/Linux/Codespaces: source venv/bin/activate

4. Install dependencies:

   pip install -r requirements.txt

## Development Workflow
Create a new branch for each specific feature or mathematical component (e.g., git checkout -b feature-lrd-calculation). Test your code locally before opening a Pull Request to the main branch.
