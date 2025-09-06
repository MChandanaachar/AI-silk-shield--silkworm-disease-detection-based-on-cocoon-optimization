## 📌 Overview
This project scrapes the latest **cocoon prices** from the [Karnataka Sericulture Department](https://karnatakasericulture.gov.in/) and displays them in a user-friendly format.  
The aim is to provide **real-time price updates** for farmers to make informed selling decisions.

## 🚀 Features
- Scrapes cocoon prices from official PDF files.
- Extracts and formats data in a clean table.
- Easy to run locally.
- Option to export results to CSV.

## 🛠 Tech Stack
- **Python 3.10+**
- `requests` – for fetching PDF files
- `pdfplumber` – for reading PDF content
- `pandas` – for data processing
- `beautifulsoup4` – for HTML parsing (if required)

## 📂 Project Structure
cocoon-price-scraper/
│-- scraper.py # Main scraping logic
│-- requirements.txt # Python dependencies
│-- README.md # Project documentation
│-- output.csv # Sample output

bash
Copy
Edit

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/cocoon-price-scraper.git
   cd cocoon-price-scraper
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
▶️ Usage
Run the scraper:

bash
Copy
Edit
python scraper.py
Output will be saved to:

lua
Copy
Edit
output.csv
📸 Screenshot

✍️ Author
M Chandana


Multilingual farmer support (Kannada, English)

👩‍💻 Author
M Chandana
AI/ML and JAVA | Passionate about Agriculture Tech and app development
