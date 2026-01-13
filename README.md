# Quantum-IDS

## Description

Quantum-IDS is an Intrusion Detection System (IDS) that leverages quantum computing concepts and quantum machine learning algorithms to enhance network security and surveillance. This project explores the application of quantum computing principles to cybersecurity, specifically using quantum kernel methods and quantum feature maps to detect network intrusions and anomalies.

The system processes network traffic data and applies quantum machine learning techniques, including Quantum Support Vector Machines (QSVM) with quantum kernels, to classify network traffic as normal or potentially malicious. By utilizing quantum feature encoding and quantum kernel computation, the project aims to demonstrate how quantum computing can potentially improve intrusion detection capabilities.

## Purpose

The primary purpose of this project is to:

- **Explore Quantum Computing Applications in Cybersecurity**: Investigate how quantum computing concepts can be applied to enhance traditional intrusion detection systems
- **Implement Quantum Machine Learning**: Build and test quantum machine learning models for network security applications
- **Enhance Network Security**: Develop a system that can potentially detect network intrusions more effectively using quantum algorithms
- **Research and Development**: Contribute to the emerging field of quantum cybersecurity by implementing and evaluating quantum-based IDS solutions

This project serves as a research initiative to bridge the gap between quantum computing and practical cybersecurity applications, demonstrating the feasibility and potential advantages of quantum-enhanced intrusion detection.

## What Was Built

This project implements a complete quantum-based intrusion detection system with the following components:

### Core Components

1. **Quantum Feature Encoding**
   - Implementation of quantum feature maps using `ZZFeatureMap` from Qiskit
   - Data preprocessing and normalization for quantum circuit compatibility
   - Principal Component Analysis (PCA) for dimensionality reduction to match quantum circuit requirements

2. **Quantum Kernel Methods**
   - Custom quantum kernel computation function that calculates similarity between data points using quantum circuits
   - Integration of quantum feature maps with quantum circuits
   - Quantum kernel matrix computation for training and testing datasets

3. **Quantum Support Vector Machine (QSVM)**
   - Implementation of QSVM using quantum kernels
   - Integration with scikit-learn's SVC using precomputed quantum kernels
   - Support for both classification and regression tasks

4. **Data Processing Pipeline**
   - Network traffic data preprocessing (from CSV format)
   - Feature extraction and normalization
   - Protocol encoding (TCP/UDP)
   - Data sampling and balancing for training

5. **Performance Evaluation System**
   - Confusion matrix visualization
   - FAR/FRR (False Acceptance Rate/False Rejection Rate) curves
   - DET (Detection Error Tradeoff) curves
   - Regression metrics (MSE, RMSE, MAE, R²)
   - Classification reports and accuracy metrics

### Technologies and Libraries Used

- **Qiskit** (v1.3.2): IBM's quantum computing framework
- **Qiskit Machine Learning**: Quantum machine learning algorithms and tools
- **Qiskit Aer**: Quantum circuit simulators
- **Qiskit IBM Runtime**: Access to IBM Quantum cloud backends
- **scikit-learn**: Classical ML algorithms and evaluation metrics
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization

### Key Features

- Quantum kernel-based classification for intrusion detection
- Support for multiple quantum backends (IBM Quantum cloud and local simulators)
- Comprehensive performance evaluation and visualization tools
- Scalable data preprocessing pipeline
- Integration of quantum and classical machine learning approaches

## How to Run/Test

### Prerequisites

1. **Python Environment**: Python 3.8 or higher
2. **IBM Quantum Account**: Sign up at [IBM Quantum](https://quantum-computing.ibm.com/) to get an API token
3. **Dataset**: Network traffic dataset in CSV format (train.csv)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Quantum-IDS
   ```

2. **Install required packages**:
   ```bash
   pip install qiskit qiskit-machine-learning qiskit-aer qiskit-ibm-runtime
   pip install scikit-learn pandas numpy matplotlib seaborn jupyterlab
   ```
   
   **Optional**: For enhanced security with `.env` files:
   ```bash
   pip install python-dotenv
   ```

3. **Configure IBM Quantum Account** (Security Best Practice):
   - Obtain your API token from [IBM Quantum](https://quantum-computing.ibm.com/)
   - **IMPORTANT**: Never hardcode tokens in code! Use environment variables instead:
     ```bash
     export IBM_QUANTUM_TOKEN="your_token_here"
     ```
   - The notebook will automatically read the token from the `IBM_QUANTUM_TOKEN` environment variable
   - The notebook includes security instructions and will raise an error if the token is not set

4. **Prepare the Dataset**:
   - Place your `train.csv` file in the appropriate directory
   - **Note**: The notebook uses `/content/train.csv` path (Google Colab default). If running locally, update the path in the notebook cells to match your local file location
   - Ensure the dataset contains network traffic features and a category/label column
   - The notebook includes code to sample and preprocess the data

### Running the Project

1. **Open Jupyter Notebook**:
   ```bash
   jupyter lab Quantum_IDS.ipynb
   ```

2. **Execute the Notebook**:
   - Run cells sequentially from top to bottom
   - The notebook includes:
     - Package installation and setup
     - IBM Quantum account configuration
     - Data loading and preprocessing
     - Quantum feature map creation
     - Quantum kernel computation
     - Model training and evaluation
     - Performance visualization

3. **Key Execution Steps**:
   - **Data Loading**: Load and preprocess the network traffic dataset
   - **Feature Engineering**: Normalize features and apply PCA for dimensionality reduction
   - **Quantum Circuit Setup**: Configure quantum feature maps and circuits
   - **Kernel Computation**: Calculate quantum kernel matrices (this may take time)
   - **Model Training**: Train the quantum SVM classifier
   - **Evaluation**: Generate predictions and evaluate model performance

### Testing

- **Local Testing**: Use Qiskit Aer simulator for local testing without requiring IBM Quantum cloud access
- **Cloud Testing**: Use IBM Quantum backends (ibm_brisbane, ibm_kyiv, ibm_sherbrooke) for cloud-based execution
- **Performance Metrics**: Review the generated confusion matrices, FAR/FRR curves, and classification reports

### Notes

- **Environment**: The notebook was originally developed for Google Colab (uses `/content/` paths). For local execution, update file paths accordingly
- Quantum kernel computation can be computationally intensive; consider using smaller datasets or sampling for initial testing
- IBM Quantum cloud access may have queue times and usage limits
- The project includes code to handle both simulated and real quantum hardware
- For local testing without IBM Quantum access, you can use Qiskit Aer simulator (no token required)

## What I Learned

Through the development of this Quantum-IDS project, I gained valuable insights and knowledge in several key areas:

### Quantum Computing Concepts

- **Quantum Circuits and Gates**: Understanding how to construct quantum circuits using Qiskit, including rotation gates (RY) and feature encoding techniques
- **Quantum Feature Maps**: Learned about different quantum feature encoding methods, particularly the ZZFeatureMap for encoding classical data into quantum states
- **Quantum Kernels**: Explored how quantum circuits can compute similarity measures (kernels) between data points, potentially offering advantages over classical kernels

### Quantum Machine Learning

- **Quantum Support Vector Machines**: Implemented QSVM using quantum kernels, understanding the integration of quantum algorithms with classical ML frameworks
- **Quantum-Classical Hybrid Approaches**: Gained experience in combining quantum and classical machine learning techniques
- **Quantum Kernel Methods**: Learned how quantum kernels are computed and used in machine learning pipelines

### Practical Implementation Skills

- **Qiskit Framework**: Mastered the use of Qiskit for quantum circuit design, simulation, and execution
- **IBM Quantum Cloud Integration**: Learned to connect and use IBM Quantum cloud services for quantum computing tasks
- **Data Preprocessing for Quantum ML**: Understood the importance of data normalization, dimensionality reduction (PCA), and feature engineering for quantum machine learning

### Cybersecurity Applications

- **Intrusion Detection Systems**: Gained knowledge about network security, traffic analysis, and intrusion detection methodologies
- **Network Traffic Analysis**: Learned about network features (ports, TCP flags, frame lengths, etc.) used in IDS
- **Performance Evaluation**: Understood security-specific metrics like FAR (False Acceptance Rate), FRR (False Rejection Rate), and DET curves

### Challenges and Solutions

- **Dimensionality Constraints**: Addressed the challenge of encoding high-dimensional classical data into quantum circuits using PCA
- **Computational Complexity**: Learned to optimize quantum kernel computations and handle large datasets efficiently
- **Quantum-Classical Integration**: Solved integration challenges between quantum algorithms and classical ML libraries (scikit-learn)

### Research Insights

- **Quantum Advantage Exploration**: Explored whether quantum computing can provide advantages in cybersecurity applications
- **Hybrid Quantum-Classical Systems**: Understood the current state of quantum machine learning and its practical limitations
- **Future Potential**: Gained perspective on the future of quantum computing in cybersecurity and network defense

This project provided hands-on experience in an emerging field at the intersection of quantum computing, machine learning, and cybersecurity, offering valuable insights into both the potential and current limitations of quantum-enhanced security systems.

---

## Authors

- **AHSSAR Hasna**
- **ATABET Nouhaila**