# CEAE
A method defense against adversarial examples attack

This repository provides the partial key implementation code for the method proposed in the paper “CEAE: An Effective Method for Handling Adversarial Examples in Modulation Recognition for Intelligent Consumer Electronics”.

The core algorithm of this work is currently under national invention patent application, and our research team is also conducting extended research based on the CEAE framework with relevant results not yet published. For the above reasons, the complete source code cannot be fully disclosed at this stage. The full codebase will be open to the academic community after the relevant research outcomes are officially published.

The code released in this repository mainly covers the core implementation of the proposed method:
The complete network architecture of the CEAE framework
Key training logic and optimization strategies adopted in the paper
Main evaluation pipelines for performance verification
With the provided core code, the defense effects of the proposed method under different adversarial attack settings can be reproduced, which is consistent with the experimental results presented in the manuscript.

Datasets
The experiments in this work are based on the following two datasets:
RML2016.10a: A widely used public benchmark dataset for automatic modulation recognition tasks.
HSDRMD: A self-built dataset by our research team, constructed to validate the generalization capability of the proposed method in more complex scenarios.
