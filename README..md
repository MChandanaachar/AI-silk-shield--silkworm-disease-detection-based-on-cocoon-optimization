
🐛 Silk Shield: AI-powered Sericulture Disease Detection and Climate-based Cocoon Optimization
📌 Overview
Silk Shield is an AI-powered tool designed to assist sericulture farmers in diagnosing silkworm diseases and optimizing cocoon pricing decisions.
The system leverages EfficientNetB3 for disease classification, scrapes daily cocoon prices from the Karnataka Sericulture Department, and provides a user-friendly interface using Streamlit/Flask.

This project aims to make disease detection faster, affordable, and accessible without relying on costly IoT devices.

✨ Features
Real-time Disease Detection 🩺
Upload an image of a silkworm and get instant disease classification.

Daily Cocoon Price Scraper 📊
Automatic scraping from the Karnataka Sericulture Department’s official site.

Farmer-Friendly Interface 💻
Easy-to-use web app powered by Streamlit/Flask.

Cost-Effective 💡
No IoT hardware required; purely AI & web-based.

🛠️ Tech Stack
Programming Language: Python

Machine Learning Model: EfficientNetB3 (Keras/TensorFlow)

Frontend: Streamlit / Flask

Web Scraping: BeautifulSoup4, Requests

Data Handling: Pandas, NumPy

Deployment: Local / Cloud (Streamlit Share, Flask Server)

📂 Project Structure
bash
Copy
Edit
SilkShield/
│── main.py                 # Entry point for the app
│── model/
│    └── efficientnetb3.h5   # Trained model weights
│── scraper/
│    └── cocoon_scraper.py   # Scrapes cocoon price data
│── utils/
│    └── preprocess.py       # Image preprocessing helpers
│── requirements.txt         # Python dependencies
│── README.md                # Project documentation
│── dataset/                 # (Optional) Dataset used for training
🚀 Installation & Usage
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/silk-shield.git
cd silk-shield
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the App
bash
Copy
Edit
# For Streamlit
streamlit run main.py

# For Flask
python main.py
📊 Dataset
Collected from real-world farm visits and public datasets.

Preprocessed to resize, normalize, and augment images for better model accuracy.

📈 Model Performance
Architecture: EfficientNetB3

Accuracy: ~95% on test set

Loss Function: Categorical Crossentropy

Optimizer: Adam

📌 Future Improvements
Multi-disease detection in a single image

Mobile app deployment

Multilingual farmer support (Kannada, Hindi, English)

👩‍💻 Author
M Chandana
AI/ML  | Passionate about Agriculture Tech and app development
