# Enhancing Charterparty Analysis: A Hybrid NLP Approach

This repository contains the source code for the hybrid NLP pipeline described in the research paper: **"Enhancing Charterparty Analysis: A Hybrid Decision Support System for Cargo-Specific Risk Profiling"**.

## 📌 Project Overview
Manual review of maritime charterparties is labor-intensive and prone to inconsistencies. This project introduces a **Hybrid NLP Architecture** that combines:
1.  **Legal-BERT:** For semantic classification of risk clauses (e.g., *Force Majeure*, *Arbitration*).
2.  **spaCy NER:** A rule-based Named Entity Recognition module acting as a regulatory compliance filter (e.g., *MARPOL*, *SOLAS*).
3.  **BERTopic:** For interpretability and clustering of clause types.

The system is designed to handle the "long-tail" distribution of risk labels in low-resource maritime settings, specifically analyzing the learnability differences between **LNG** and **Tanker** segments.

## 🚀 Repository Structure
* `hybrid_pipeline.py`: The main Python script containing the model training, NER constraints, and evaluation logic.
* `requirements.txt`: List of dependencies required to run the code.

## 🛠️ Installation & Usage
To reproduce the experimental results:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

2.  **Run the Pipeline:**
    ```bash
    python hybrid_pipeline.py
    ```

## 📊 Methodology
The model utilizes a **class-weighted Binary Cross-Entropy (BCE) loss** to mitigate label imbalance and integrates expert-defined constraints to ensure regulatory compliance detection.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
