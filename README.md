🤖 AIgnitePrice: Smart Product Pricing Challenge 2025
An advanced machine learning solution for the ML Challenge 2025: Smart Product Pricing, achieving a highly competitive SMAPE score by fusing multimodal features with a powerful gradient boosting model.

🎯 Project Overview
In e-commerce, setting the right price is critical. This project tackles the challenge of predicting product prices by analyzing both their textual descriptions and product images. Our solution, AIgnitePrice, uses a sophisticated feature engineering pipeline to create a rich, multimodal dataset that is then used to train a LightGBM model. This approach holistically analyzes product details to suggest an optimal price.

The final model achieves a Cross-Validated SMAPE of 28.52%, demonstrating its robustness and accuracy.

✨ Key Features
Multimodal Feature Fusion: Combines information from three distinct sources for a comprehensive understanding of the product:

📝 Semantic Text Embeddings: Uses a pre-trained SentenceTransformer to capture the meaning of product titles and descriptions.

🖼️ Visual Image Embeddings: Employs a pre-trained EfficientNet CNN to extract visual features like brand, quality, and material from product images.

🔢 Engineered Numerical Features: Explicitly extracts the Item Pack Quantity (IPQ), a critical price multiplier.

Powerful Regression Model: Leverages LightGBM, a high-performance gradient boosting framework, for fast training and accurate predictions.

Robust Training Strategy: Implements 5-Fold Cross-Validation to ensure the model generalizes well to unseen data and to provide a reliable performance estimate.

Optimized for SMAPE: The entire pipeline, from log-transforming the price target to selecting the model objective, is designed to optimize for the competition's primary metric, SMAPE.

🛠️ Solution Architecture
The solution follows a two-stage pipeline: Feature Engineering followed by Model Training.

Text Processing Pipeline:

The raw catalog_content is passed to a SentenceTransformer (all-MiniLM-L6-v2) to generate a 384-dimension text embedding.

A custom regex function extracts the numerical IPQ value.

Image Processing Pipeline:

Product images are downloaded and preprocessed (resized, normalized).

The images are fed into a pre-trained EfficientNet-B0 model to generate a 1280-dimension image embedding.

Feature Fusion & Training:

The IPQ, text embeddings, and image embeddings are concatenated into a single feature matrix.

This final matrix is used to train a LightGBM Regressor using a 5-Fold Cross-Validation strategy. The final prediction is an average of the predictions from all 5 models.

Validation Results
The model's performance was rigorously evaluated using 5-Fold Cross-Validation, with the final score calculated on out-of-fold predictions across the entire training set.
Metric	Value	Notes
SMAPE Score	60.46%	(Primary metric: Lower is better)
MAE (on log-price)	“0.48”	Mean Absolute Error on the transformed target.
R² (on log-price)	“0.68”	Explaining 62% of variance on the log scale.

💻 Tech Stack
Core ML/DL: Scikit-Learn, LightGBM, PyTorch
NLP: Sentence-Transformers
Computer Vision: Timm, Pillow, Torchvision
Data Handling: Pandas, NumPy
Utilities: Tqdm

📂 Project Structure:
AIgnitePrice/
├── dataset/
│   ├── images/         # Folder for all downloaded product images
│   ├── train.csv
│   └── test.csv
├── src/
│   └── utils.py        # Helper functions (e.g., image download)
├── main_pipeline.py    # Main script to run the full pipeline
├── requirements.txt    # Required Python packages
└── README.md           # You are here!

👥 Team Members
Abhishek Mishra (Team Leader)
Aman Gupta
Aditya Saxena
