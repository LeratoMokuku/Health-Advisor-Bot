# Health-Advisor-Bot
Health Advisor Bot is an AI-powered diagnostic system using TF-IDF/cosine similarity to match symptoms with conditions. It provides medicine, herbal, nutrition &amp; lifestyle recommendations. Demonstrates genAI understanding through intelligent corpus construction &amp; multi-modal responses for African healthcare contexts.

 🩺 Health Advisor Bot

An AI-powered health advisory system that provides symptom-based health recommendations using machine learning and comprehensive medical datasets. Built for the African context with support for conventional medicine, herbal treatments, and nutritional advice.

🌟 Features

- Symptom-Based Diagnosis: Input symptoms and receive potential condition matches
- Multi-Modal Recommendations: 
  - 💊 Conventional medicine suggestions
  - 🌿 Herbal and traditional remedies  
  - 🍎 Nutritional and dietary advice
  - ❤️ Lifestyle recommendations

run jupyternotebook
   run_advisor_interactive()
   
   web interface
Prerequisites
- Python 3.8+
- pip package manager

 Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd health-advisor
   
2. Install dependencies

pip install pandas scikit-learn flask


Add your datasets (place in project root):

    symptoms.csv - Primary symptom-disease mappings

    observations.csv - Clinical observations and causes

    conditions.csv - Medical conditions database

    careplans.csv - Treatment plans and recommendations

    medicine_disease.csv - Medicine recommendations

    herbaltreatment_disease.csv - Herbal remedies

    nutrition_disease.csv - Nutritional advice

    allergies.csv - Allergy information
    
 Run the application
   python app.py
   
Access the web interface
   Open http://localhost:5000 in your browser
