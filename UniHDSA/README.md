# UniHDSA: A Unified Relation Prediction Approach for Hierarchical Document Structure Analysis

## Introduction

Document structure analysis is essential for understanding both the physical layout and logical structure of documents, aiding in tasks such as information retrieval, document summarization, and knowledge extraction. Hierarchical Document Structure Analysis (HDSA) aims to restore the hierarchical structure of documents created with hierarchical schemas. Traditional approaches either focus on specific subtasks in isolation or use multiple branches to address distinct tasks. In this work, we introduce UniHDSA, a unified relation prediction approach for HDSA that treats various subtasks as relation prediction problems within a consolidated label space. This allows a single module to handle multiple tasks simultaneously, improving efficiency, scalability, and adaptability. Our multimodal Transformer-based system demonstrates state-of-the-art performance on the Comp-HRDoc benchmark and competitive results on the DocLayNet dataset, showcasing the effectiveness of our method across all subtasks.

## Reproduction

This project is built on [detrex](https://github.com/IDEA-Research/detrex/tree/main), a library for computer vision. Due to company policy, we cannot release the code for the model. However, we provide the detailed configuration including the model architecture, training hyperparameters, and data processing methods. We also provide the code for the evaluation of the model.