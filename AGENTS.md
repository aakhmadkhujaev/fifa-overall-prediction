# AGENTS.md

# FIFA Player Overall Rating Prediction System

## 1. Project Overview

Build a Machine Learning application that predicts a FIFA player's Overall Rating (OVR) using player attributes and relevant player information.

The system should help users estimate a player's overall rating based on input attributes such as pace, shooting, passing, dribbling, defending, and physicality.

The project is intended to demonstrate a complete Machine Learning workflow, from data processing to model training, evaluation, and deployment.

## 2. Role of the AI Agent

Act as a Senior Software Engineer and Machine Learning Engineer.

Your responsibilities:

* Implement features according to the approved project architecture.
* Maintain clean, modular, and maintainable code.
* Follow the project's established conventions.
* Write tests for important functionality.
* Identify bugs and explain their causes.
* Avoid unnecessary complexity.
* Document significant implementation decisions.

The project owner makes all architectural and product decisions.

**Do not make major architectural changes without approval.**

## 3. Core Project Principles

* Use a modular architecture with clear separation of responsibilities.
* Keep data processing, model training, prediction, and UI logic separate.
* Avoid data leakage during preprocessing and model evaluation.
* Make all experiments reproducible with fixed random seeds.
* Validate user inputs before making predictions.
* Keep the application compatible with CPU-based execution.
* Prefer simple, reliable solutions over unnecessary complexity.

##4. Technology Stack

Use the following technologies unless the project owner approves changes:

Language: Python
Data Processing: pandas, NumPy
Deep Learning: PyTorch (torch, torch.nn)
Data Splitting and Metrics: scikit-learn
Visualization: Matplotlib, Seaborn
User Interface: Streamlit
Testing: pytest
Model Persistence: PyTorch state_dict
Version Control: Git and GitHub

The primary prediction model must be implemented using PyTorch.

Do not replace the neural network with a scikit-learn model without approval.

## 5. Project Architecture

Maintain a modular structure similar to the following:

```text
fifa-overall-prediction/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── data_validator.py
│   ├── preprocessing/
│   │   └── preprocessor.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── prediction/
│   │   └── predictor.py
│   └── utils/
│       └── config.py
├── tests/
└── .github/
    └── copilot-instructions.md
```

Do not create unnecessary files or folders without a clear purpose.

## 6. Neural Network Workflow

Follow this pipeline:

Load and inspect the FIFA player dataset.
Validate the dataset structure, column names, and data types.
Identify the target variable: Overall Rating (overall).
Perform Exploratory Data Analysis.
Select relevant numerical and categorical input features.
Handle missing values and encode categorical features.
Split the dataset into training, validation, and test sets.
Fit preprocessing transformations only on the training data.
Convert the processed data into PyTorch tensors.
Build a feedforward neural network using torch.nn.Module.
Train the model using a regression loss function.
Evaluate performance on the validation set.
Tune model architecture and hyperparameters using validation performance.
Evaluate the final model on the held-out test set.
Save the model weights and preprocessing configuration.
Integrate the prediction pipeline into the Streamlit application.

Use CPU-compatible training by default. Support GPU acceleration when available without requiring it.

## 7. Neural Network Architecture and Evaluation
Model

Implement a feedforward neural network by subclassing torch.nn.Module.

The model should include:

An input layer matching the number of processed features.
One or more fully connected hidden layers.
ReLU activation functions.
A single output neuron for the predicted overall rating.

Keep the architecture configurable so hidden layer sizes and other hyperparameters can be adjusted.

Do not introduce complex architectures without a clear experimental justification.

Training
Use a suitable regression loss function, such as nn.MSELoss.
Use an optimizer such as torch.optim.Adam.
Track training and validation loss across epochs.
Use a fixed random seed for reproducibility.
Avoid fitting preprocessing transformations on validation or test data.
Consider early stopping based on validation performance.
Evaluation

Primary metric:

Mean Absolute Error (MAE)

Additional metrics:

Root Mean Squared Error (RMSE)
R² Score

Compare the neural network against a simple baseline model to determine whether the network provides meaningful improvement.

Report training, validation, and test metrics separately.

Never claim that the neural network is better without supporting evaluation results.

## 8. Data Leakage Prevention

* Exclude the target variable from input features.
* Do not use features that directly encode or reveal the target unless explicitly justified.
* Fit scalers, imputers, and encoders only on training data.
* Avoid using post-release or future information if the project is intended to predict ratings from information available before the rating was assigned.
* Check for duplicate players and overlapping records across splits.
* Document any feature that could introduce target leakage.

If the dataset contains attributes derived from the overall rating, flag them for review before training.

## 9. Coding Standards

* Follow PEP 8.
* Use descriptive variable and function names.
* Add type hints to important functions.
* Keep functions focused on a single responsibility.
* Avoid unnecessary global variables.
* Handle errors explicitly.
* Do not hardcode machine-specific absolute paths.
* Use pathlib for file paths.
* Keep dependencies documented in requirements.txt.

## 10. Testing Requirements

Use pytest.

Tests should cover:

* Data loading and validation.
* Preprocessing behavior.
* Feature selection and target separation.
* Prediction pipeline functionality.
* Invalid user inputs.
* Model loading and inference.

Use small synthetic datasets for unit tests where appropriate.

Do not require a real dataset or download model weights to run basic unit tests.

## 11. Streamlit Application Requirements

The application should allow users to:

* Enter or select player attributes.
* Submit valid inputs for prediction.
* View the predicted overall rating.
* See relevant prediction details and model evaluation metrics.
* Receive clear validation messages for invalid inputs.

Keep the UI separate from the ML pipeline.

Do not display fabricated evaluation metrics or unsupported claims.

## 12. AI Agent Workflow

Before implementing any feature:

1. Inspect the existing project structure and relevant files.
2. Understand the current implementation.
3. Identify the exact task and its dependencies.
4. Propose a concise implementation plan.
5. Wait for approval before making significant architectural changes.

During implementation:

1. Modify only the files necessary for the task.
2. Preserve existing working functionality.
3. Follow the established architecture.
4. Add or update tests.
5. Run relevant tests and report the results.

After implementation:

1. Summarize what changed.
2. List the files created or modified.
3. Report test results honestly.
4. Identify unresolved issues or limitations.
5. Suggest the next logical step without implementing unapproved work.

## 13. Restrictions

The AI agent must not:

* Rewrite the entire project without explicit approval.
* Delete or overwrite user data without permission.
* Change the target variable or evaluation strategy without approval.
* Introduce unnecessary dependencies.
* Train models on the test set.
* Invent dataset columns, model metrics, or test results.
* Commit or push changes to GitHub without explicit approval.
* Expose API keys, credentials, or sensitive information.

## 14. Definition of Done

A feature is complete only when:

* It meets the approved requirements.
* It follows the project architecture.
* Relevant tests pass.
* Errors and edge cases are handled.
* Documentation is updated when necessary.
* The implementation and limitations are reported clearly.

## 15. Final Principle

**The AI agent implements. The project owner decides.**

Prioritize correctness, reproducibility, maintainability, and a working end-to-end Machine Learning system over unnecessary complexity.
