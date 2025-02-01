Sarcasm Detection Using Machine Learning 🤖

Project Overview

This project aims to build an intelligent sarcasm detection system using machine learning techniques. Given the complexity of sarcasm, where textual meaning often contradicts the speaker’s intent, the system classifies text inputs as sarcastic or non-sarcastic. It has practical applications in sentiment analysis, content moderation, and natural language processing tasks.

Key Features 🔑


Single Sentence Detection:
Detect sarcasm in a single sentence through real-time input.

Batch Detection:
Upload text files for batch processing and receive detailed sarcasm probability results.

Interactive User Interface:
Built using Flask, the web interface allows users to input text or upload files seamlessly.

Model Accuracy:
Achieved 83% accuracy with balanced precision, recall, and F1-scores.

Technologies Used 📡


Machine Learning: Logistic Regression for text classification

Data Handling: Vectorization with bag-of-words and data balancing using SMOTE (Synthetic Minority Oversampling Technique)

Evaluation Metrics: Accuracy, precision, recall, and F1-score

Backend: Flask for model deployment and API integration

Frontend: Bootstrap for a responsive user interface


Getting Started 🎬


Clone the repository:
git clone <repository-url>

Navigate to the project directory and install dependencies:
pip install -r requirements.txt

Run the Flask application:
python app.py

Access the interface at http://localhost:5000/


How It Works ⚙️

Input text or upload a file

Preprocessing: Text is vectorized and passed through the logistic regression model

Prediction: The system returns sarcasm probabilities

Output: Clear and visually appealing results displayed on the interface

Future Scope 🔮

* Incorporate advanced models like transformers for more nuanced sarcasm detection
* Expand the dataset to enhance generalization
* Build a comprehensive API for broader integration
* Contributing


Feel free to fork the repository and submit pull requests to improve the project further.


License: This project is licensed under the MIT License.
