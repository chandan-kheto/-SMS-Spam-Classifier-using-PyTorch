📩 SMS Spam Classifier using PyTorch
A Deep Learning-based project to classify SMS messages as Spam or Ham (Genuine) using PyTorch.
This project is my first step into Deep Learning and Natural Language Processing (NLP), inspired by real-world spam filtering systems like Gmail, WhatsApp, and banking alerts.

🚀 Features
Preprocesses raw SMS text data (cleaning, tokenization, encoding)

Uses a Neural Network built with PyTorch

Classifies messages into Spam or Ham

Trains & evaluates model performance

Can be extended to email, social media, or chat spam filtering

📂 Dataset
The dataset used is the SMS Spam Collection Dataset from Kaggle:
🔗 Dataset Link

It contains:

label: "spam" or "ham"

message: The SMS text content

🛠 Technologies Used
Python 3.10+

PyTorch (Deep Learning Framework)

Pandas & NumPy (Data Handling)

Scikit-learn (Train-Test Split & Metrics)

Matplotlib/Seaborn (Visualization)

📊 Project Workflow
Load Dataset

Data Preprocessing

Remove extra spaces, punctuation, and special characters

Tokenize text

Convert words to numeric indices

Model Building (PyTorch)

Define input, hidden, and output layers

Use activation functions like ReLU & Sigmoid

Training

Loss function: Binary Cross Entropy Loss

Optimizer: Adam

Evaluation

Accuracy

Confusion Matrix

Prediction on New Messages

📌 Example Usage

# Sample prediction
message = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
predict(message)  # Output: SPAM

message = "Are we still meeting for lunch today?"
predict(message)  # Output: HAM
📈 Results
Accuracy: 91.12% on test set

Model detects spam messages with high precision and recall.

💡 Real-World Applications
Email spam filtering

WhatsApp/Telegram message classification

Banking fraud SMS detection

Customer support automation
