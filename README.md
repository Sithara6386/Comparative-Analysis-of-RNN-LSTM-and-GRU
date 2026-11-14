# Problem 1: Comparative-Analysis-of-RNN-LSTM-and-GRU

A comparative study of Vanilla RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) networks for sequence learning, time-series classification, and long-term dependency modelling. This project implements and evaluates three recurrent neural network architectures using PyTorch. 

Two primary tasks are considered:


     a) Human Activity Recognition (HAR): Classifying human activities using multivariate sensor data.
   
     b) Adding Problem: Testing the ability of recurrent networks to remember two marked values across long sequences
     and predict their sum.

 
   Custom recurrent cells are implemented from scratch using matrix operations and verified against PyTorch's built-in    recurrent layers.


       Project Objectives

       a) Implement MyRNNCell, MyLSTMCell and MyGRUCell from scratch.
       b) Verify custom implementations against PyTorch reference implementations.
       c) Build a subject-wise HAR data pipeline with training-only normalisation.
       d) Compare RNN, LSTM and GRU classification performance.
       e) Evaluate sequence learning across different sequence lengths.
       f) Investigate vanishing gradients and long-term memory.
       g) Perform hidden-size and gradient-clipping ablation studies.
       h) Identify a suitable architecture for resource-constrained wearable devices.


Parameter counts
For input dimension 9 and hidden size 64:


Experimental Configuration

     1) Framework: PyTorch
     2) Optimiser: Adam
     3) Learning rate: 0.001
     4) Batch size: 64
     5) Main hidden size: 64
     6) Random seeds: 42, 123 and 2026
     7) HAR sequence length: 128
     8) Adding-problem sequence lengths: 50, 100, 200 and 400
     9) Main training budget: 30 epochs
     10) Gradient clipping threshold: 1.0
   
All three architectures are evaluated under consistent experimental conditions within each experiment.

Experiments

1. Data Pipelines
   
           Subject-wise validation split for HAR.
           Channel-wise mean and standard deviation calculated using training data only.
           Class distribution visualisation.
           Adding-problem generator validation.
           Verification that each sequence contains exactly two markers, one in each half.

2. Custom Recurrent Cells
   
          MyRNNCell with tanh activation.
          MyLSTMCell with input, forget, output and candidate gates.
          MyGRUCell with update and reset gates.
          Weight copying and numerical comparison against PyTorch implementations.

3. HAR Classification: Three recurrent classifiers process 128 time steps and use the final hidden state for classification.

     Evaluation metrics:

          Test accuracy.
          Macro-F1 score.
          Model parameter count.
          Seconds per epoch.
          Confusion matrix.
   
   Three random seeds are used for each architecture.

4. Adding Problem: Models are trained using sequence lengths of 50, 100, 200 and 400.

     Evaluation includes:

         Test mean squared error (MSE).
         Mean and standard deviation across seeds.
         Training and validation learning curves.
         Number of successful runs.

   A run is considered solved when test MSE is below 0.05. The constant-prediction baseline is approximately 0.167.

5. Gradient and Ablation Analysis
   
        Gradient norm visualisation before and after training at T=100.
        Hidden-size sweep for LSTM and GRU: 16, 32, 64 and 128.
        Gradient clipping ablation for vanilla RNN at T=200.
        Comparison of accuracy, parameter count and computational time.

# Problem 2: Transformer Implementation and Experiments

1. Overview

This project implements and evaluates Transformer-based architectures using PyTorch. It explores scaled dot-product attention, Transformer encoder blocks, Human Activity Recognition (HAR) classification, and character-level language modelling using the Tiny Shakespeare dataset.

The objective is to understand the Transformer architecture through implementation from scratch, unit testing, model training, and experimental analysis.


Task A: Implemented scaled dot-product attention from scratch with optional masking.

            Query, Key and Value projections.
            scaled attention score calculation.
            Softmax-based attention weights.
            Optional attention masking.
            Causal masking to prevent access to future token


Task B: Implemented a Transformer encoder using:

          Multi-head self-attention.
          Pre-Layer Normalization.
          Feed-forward neural network.
          Residual connections.
          Positional encodings.

Task C: The Transformer encoder was applied to the UCI Human Activity Recognition (HAR) dataset.

          Dataset: UCI HAR.
          Input features: 561 engineered features.
          Patch size: 8.
          Number of tokens: 70.
          Embedding dimension: 128.
          Transformer layers: 4.
          Positional encoding: No positional encoding, sinusoidal encoding and learned encoding.

Task D: character-level language model was implemented using the Tiny Shakespeare dataset.

          Character-level tokenisation.
          Token embeddings.
          Positional embeddings.
          Causal self-attention.
          Transformer decoder blocks.
          Autoregressive text generation.

Task E: Experimental Analysis and Ablation

          Comparison of different numbers of attention heads.
          Comparison of Transformer performance with and without causal masking.
          Analysis of attention heatmaps.
          Training time and GPU memory analysis across different context lengths.
          Key-Value (KV) cache performance evaluation.
          Comparison with an LSTM character-level language mode


# Execution Environment

The experiments were conducted using:

       Component: Configuration
       Platform: Google Colab
       Programming Language: Python
       Deep Learning Framework: PyTorch
       Hardware Accelerator: NVIDIA Tesla T4 GPU
       Notebook Environment: Google Colab Jupyter Notebook


Run time: Problem 1: 5 Minutes,  Problem 2:15 Minutes 
