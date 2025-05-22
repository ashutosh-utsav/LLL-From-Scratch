# Large Language Model from Scratch (GPT)

The goal of this project is to **gain a deep understanding of how LLMs work** by building one from the ground up, focusing on **data handling, math, and the transformer architecture** .


## Features & Concepts Covered

*   **Building from Scratch:** Implementing core components of an LLM .
*   **Local Computation:** The project uses local computation and **does not require paid datasets or cloud computing**.
*   **Bigram Language Model:** Starting with a simpler model to understand fundamental concepts.
*   **Data Handling:** Processing and managing text data. Initial examples use small datasets like the 100Year of solitude. 
*   **PyTorch:** Utilizing the PyTorch library for building neural networks . Includes exploring basic PyTorch functions, tensors, and GPU acceleration.
*   **Neural Network Fundamentals:** Concepts like learnable parameters (NN.Module, NN.Linear), forward pass, optimizers (AdamW), and loss functions (Cross Entropy) are implemented.
*   **Training Process:** Implementing training loops, understanding gradient descent, and periodically estimating/reporting loss.
*   **GPT Architecture:** Building a decoder-only transformer architecture. Key components include:
    *   Token Embeddings and Positional Embeddings (learnable) .
    *   Decoder Blocks containing Multi-Head Self-Attention and Feed-Forward Networks.
    *   Self-Attention mechanism with Keys, Queries, and Values .
    *   Scaled Dot Product Attention and Masked Self-Attention .
    *   Layer Normalization and Residual Connections (Post-norm architecture).
    *   Dropout for preventing overfitting .
*   **Hyperparameters:** Understanding key parameters like `block_size`, `batch_size`, `n_layer`, `n_head`, `n_embed`, `learning_rate`, etc..
*   **Model Saving and Loading:** Implementing functionality to save trained model parameters and load them later for continued training or inference.
*   **Chatbot Interface:** A simple script to interact with the trained model for text generation (prompt completion). Includes handling context size using cropping.
*   **Introduction to Advanced Concepts:** Brief mentions of topics like Fine-tuning, Quantization , and resources like Hugging Face.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:ashutosh-utsav/LLL-From-Scratch.git
    cd LLL-From-Scratch
    ```
2.  **Create a virtual environment:** Using `venv` or Anaconda's `conda`. 
    ```bash
    virtualenv .envllm 
    # or for Anaconda: conda create -n CUDA python=<python_version>
    ```
3.  **Activate the virtual environment:**
    *   On Windows: `.\CUDA\Scripts\activate`
    *   On macOS/Linux: `source .enllm/bin/activate`
4.  **Install dependencies:**
    ```bash
    pip install -r req.txt
    ```
5.  **Data:**
    *   For initial testing, the `100YearOfSolitude.txt` file might be included in the repository.
    *   To train on larger data, you'll need to download and preprocess the OpenWebText corpus.

