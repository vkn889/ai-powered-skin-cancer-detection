IMPORTANT NOTE: DATASET CAN BE RETRIVED HERE (NOTE: NOT INCLUDED IN THE REPOSITORY AS THE FILE SIZE IS TO LARGE TO INCLUDE) https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

AI-Powered Skin Cancer Detection System
An intelligent medical imaging application that uses deep learning and computer vision to classify skin lesions as benign or malignant, potentially assisting in early skin cancer screening.
Project Overview
This project implements a binary classification system for skin cancer detection using transfer learning with the VGG16 convolutional neural network. The system processes dermoscopic images of skin lesions and provides real-time predictions through an intuitive web interface built with Streamlit.
Key Features

Deep Learning Classification: VGG16 transfer learning model for accurate skin lesion analysis
Real-time Predictions: Instant classification with confidence scores
User-friendly Interface: Streamlit web application for easy image upload and analysis
Medical Image Processing: Advanced computer vision preprocessing pipeline
High Accuracy: Achieved 87.77% validation accuracy on test dataset
Balanced Dataset: Implemented undersampling techniques to handle class imbalance

Technologies Used

Deep Learning: TensorFlow/Keras, VGG16 Transfer Learning
Computer Vision: OpenCV, PIL (Python Imaging Library)
Data Science: NumPy, Pandas, Matplotlib, Scikit-learn
Web Framework: Streamlit
Model Deployment: Joblib model serialization
Development Environment: Google Colab, Jupyter Notebook

Dataset

Source: Kaggle skin cancer dataset with dermoscopic images
Classes: Binary classification (Benign vs Malignant)
Preprocessing: Image resizing to 224x224 pixels, normalization, class balancing
Training Split: 67% training, 33% validation
Total Samples: ~3,000 balanced samples

Model Architecture
VGG16 Base Model (Pre-trained on ImageNet)
    ↓
Global Average Pooling 2D
    ↓
Dense Layer (1024 neurons, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense Layer (512 neurons, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (1 neuron, Sigmoid)
Performance Metrics

Validation Accuracy: 87.77%
Training Accuracy: 99.36%
Loss Function: Binary Crossentropy
Optimizer: SGD with momentum (learning_rate=1e-4, momentum=0.95)
Training Epochs: 30

Installation & Usage
Prerequisites
bashpip install streamlit tensorflow opencv-python numpy pillow joblib matplotlib
Running the Application
bash# Start the Streamlit web app
streamlit run app.py
Using the Model

Upload a skin lesion image (JPG, PNG formats supported)
Wait for AI processing (typically 2-3 seconds)
View prediction result with confidence score
Interpret results with medical disclaimer guidance

Project Structure
skin-cancer-detection/
├── app.py                          # Streamlit web application
├── vgg_model.joblib                 # Trained model file
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── training_notebook.ipynb          # Model training code
└── sample_images/                   # Test images for demo
Technical Implementation
Data Preprocessing Pipeline

Image Loading: OpenCV-based image reading and decoding
Resizing: Standardization to 224x224 pixel resolution
Normalization: Pixel value scaling to [0,1] range
Class Balancing: Undersampling majority class for dataset balance

Model Training Process

Transfer Learning: Leveraged pre-trained VGG16 weights
Fine-tuning: Added custom classification layers
Regularization: Implemented dropout layers to prevent overfitting
Optimization: SGD optimizer with momentum for stable convergence

Web Application Features

File Upload: Support for multiple image formats
Real-time Processing: Instant image classification
Visual Feedback: Image display with prediction overlay
Confidence Scoring: Probability-based result interpretation
Medical Disclaimer: Appropriate healthcare guidance

Medical Disclaimer
This application is designed for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.
Learning Outcomes

Deep Learning: Hands-on experience with CNN architectures and transfer learning
Computer Vision: Image preprocessing and medical imaging techniques
Web Development: Full-stack application deployment with Streamlit
Data Science: Dataset handling, class imbalance, and model evaluation
Healthcare AI: Understanding of medical AI ethics and limitations

Future Enhancements

 Multi-class classification (melanoma subtypes)
 Data augmentation for improved generalization
 Model interpretability with grad-CAM visualizations
 Integration with DICOM medical imaging standards
 Mobile application development
 Real-time camera capture functionality

Contributing
This project was developed as part of the Inspirit AI program. Contributions, suggestions, and improvements are welcome through pull requests and issues.
License
This project is available under the MIT License. See LICENSE file for details.
Acknowledgments

Inspirit AI: Educational program framework and guidance
Kaggle: Skin cancer dataset provision
TensorFlow/Keras: Deep learning framework
Streamlit: Web application framework
Medical AI Community: Research and best practices in healthcare applications


Built with dedication and AI for advancing healthcare technology
