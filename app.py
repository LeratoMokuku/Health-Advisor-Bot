
# app.py – Health Advisor (COMPREHENSIVE CORPUS APPROACH)

import os
import re
import unicodedata
import logging
from collections import defaultdict
from flask import Flask, request, jsonify, send_from_directory

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- CONFIG -------------------
DATA_DIR = "."                     # CSVs live next to app.py
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
# ---------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("health-advisor")

# ---------- LOAD CSVs (safe) ----------
files = {
    "symptoms": "symptoms.csv",
    "allergies": "allergies.csv",
    "conditions": "conditions.csv",
    "observations": "observations.csv",
    "careplans": "careplans.csv",
    "immunizations": "immunizations.csv",
    "procedures": "procedures.csv",
    "medicine": "medicine_disease.csv",
    "herbal": "herbaltreatment_disease.csv",
    "nutrition": "nutrition_disease.csv",
}

datasets = {}
for name, f in files.items():
    p = os.path.join(DATA_DIR, f)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, dtype=str).fillna("")
            datasets[name] = df
            log.info("Loaded %s – %d rows, columns: %s", name, len(df), list(df.columns))
        except Exception as e:
            log.error("CSV load error %s: %s", f, e)
    else:
        log.warning("Missing CSV: %s", f)

# ---------- TEXT CLEAN ----------
def clean(s):
    if not isinstance(s, str): s = str(s)
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- BUILD COMPREHENSIVE CORPUS ----------
corpus = []

log.info("Building comprehensive corpus with intelligent priority system...")

# === PRIORITY 1: SYMPTOMS.CSV (Primary - Most Valuable) ===
if 'symptoms' in datasets:
    sym_df = datasets['symptoms']
    disease_col = 'Disease'
    symptoms_col = 'Symptoms'
    
    symptom_entries = 0
    for i, r in sym_df.iterrows():
        disease = str(r.get(disease_col, '')).strip()
        symptoms = str(r.get(symptoms_col, '')).strip()
        
        if disease and symptoms:
            # Clean and format symptoms
            clean_symptoms = symptoms.replace(';', ' ').replace(',', ' ')
            text = f"symptoms: {clean_symptoms}"
            corpus.append({
                'id': f'sym_{i}', 
                'type': 'symptoms', 
                'text': text, 
                'diagnosis': disease,
                'symptoms': clean_symptoms,
                'priority': 1
            })
            symptom_entries += 1
    log.info("✓ Added %d symptom-disease mappings", symptom_entries)

# === PRIORITY 2: OBSERVATIONS (Clinical Context) ===
if 'observations' in datasets:
    obs_df = datasets['observations']
    obs_col = next((c for c in obs_df.columns if 'obser' in c.lower() or 'symptom' in c.lower() or 'description' in c.lower()), obs_df.columns[0])
    cause_col = next((c for c in obs_df.columns if 'cause' in c.lower() or 'possible' in c.lower() or 'diagnosis' in c.lower()), None)
    
    observation_entries = 0
    for i, r in obs_df.iterrows():
        obs = str(r.get(obs_col, '')).strip()
        cause = str(r.get(cause_col, '')).strip() if cause_col else ''
        
        if obs:
            text = f"observation: {obs}"
            if cause:
                text += f" possible_cause: {cause}"
            
            diagnosis = cause if cause else obs
            corpus.append({
                'id': f'obs_{i}', 
                'type': 'observation', 
                'text': text, 
                'diagnosis': diagnosis,
                'observation': obs,
                'cause': cause,
                'priority': 2
            })
            observation_entries += 1
    log.info("✓ Added %d clinical observations", observation_entries)

# === PRIORITY 3: CONDITIONS (Structured Medical Knowledge) ===
if 'conditions' in datasets:
    cond_df = datasets['conditions']
    cond_col = next((c for c in cond_df.columns if 'cond' in c.lower() or 'condition' in c.lower() or 'diagnosis' in c.lower()), cond_df.columns[0])
    
    # Look for additional context columns
    desc_col = next((c for c in cond_df.columns if 'desc' in c.lower() or 'description' in c.lower()), None)
    severity_col = next((c for c in cond_df.columns if 'sever' in c.lower()), None)
    age_col = next((c for c in cond_df.columns if 'age' in c.lower()), None)
    gender_col = next((c for c in cond_df.columns if 'gender' in c.lower()), None)
    
    condition_entries = 0
    for i, r in cond_df.iterrows():
        condition = str(r.get(cond_col, '')).strip()
        description = str(r.get(desc_col, '')).strip() if desc_col else ''
        severity = str(r.get(severity_col, '')).strip() if severity_col else ''
        age = str(r.get(age_col, '')).strip() if age_col else ''
        gender = str(r.get(gender_col, '')).strip() if gender_col else ''
        
        if condition:
            text_parts = [f"condition: {condition}"]
            if description:
                text_parts.append(f"description: {description}")
            if severity:
                text_parts.append(f"severity: {severity}")
            if age:
                text_parts.append(f"age: {age}")
            if gender:
                text_parts.append(f"gender: {gender}")
            
            text = " ".join(text_parts)
            corpus.append({
                'id': f'cond_{i}', 
                'type': 'condition', 
                'text': text, 
                'diagnosis': condition,
                'description': description,
                'priority': 3
            })
            condition_entries += 1
    log.info("✓ Added %d medical conditions", condition_entries)

# === PRIORITY 4: CAREPLANS (Treatment Context) ===
if 'careplans' in datasets:
    cp_df = datasets['careplans']
    diag_col = next((c for c in cp_df.columns if 'diag' in c.lower() or 'condition' in c.lower()), cp_df.columns[0])
    plan_col = next((c for c in cp_df.columns if 'care' in c.lower() or 'plan' in c.lower() or 'treatment' in c.lower()), None)
    goal_col = next((c for c in cp_df.columns if 'goal' in c.lower() or 'objective' in c.lower()), None)
    
    careplan_entries = 0
    for i, r in cp_df.iterrows():
        diag = str(r.get(diag_col, '')).strip()
        plan = str(r.get(plan_col, '')).strip() if plan_col else ''
        goal = str(r.get(goal_col, '')).strip() if goal_col else ''
        
        if diag:
            text_parts = [f"diagnosis: {diag}"]
            if plan:
                text_parts.append(f"care_plan: {plan}")
            if goal:
                text_parts.append(f"treatment_goal: {goal}")
            
            text = " ".join(text_parts)
            corpus.append({
                'id': f'cp_{i}', 
                'type': 'careplan', 
                'text': text, 
                'diagnosis': diag,
                'care_plan': plan,
                'priority': 4
            })
            careplan_entries += 1
    log.info("✓ Added %d care plans", careplan_entries)

# === PRIORITY 5: ALLERGIES (Additional Medical Context) ===
if 'allergies' in datasets:
    allergy_df = datasets['allergies']
    allergy_col = next((c for c in allergy_df.columns if 'aller' in c.lower() or 'reaction' in c.lower()), allergy_df.columns[0])
    severity_col = next((c for c in allergy_df.columns if 'sever' in c.lower()), None)
    
    allergy_entries = 0
    for i, r in allergy_df.iterrows():
        allergy = str(r.get(allergy_col, '')).strip()
        severity = str(r.get(severity_col, '')).strip() if severity_col else ''
        
        if allergy:
            text = f"allergy: {allergy}"
            if severity:
                text += f" severity: {severity}"
            
            corpus.append({
                'id': f'allergy_{i}', 
                'type': 'allergy', 
                'text': text, 
                'diagnosis': f"Allergy: {allergy}",
                'priority': 5
            })
            allergy_entries += 1
    log.info("✓ Added %d allergy entries", allergy_entries)

# === DEDUPLICATION AND QUALITY CONTROL ===
log.info("Performing corpus optimization...")

# Remove exact duplicates
initial_count = len(corpus)
seen_texts = set()
unique_corpus = []

for item in corpus:
    clean_t = clean(item['text'])
    if clean_t not in seen_texts and clean_t.strip():
        seen_texts.add(clean_t)
        unique_corpus.append(item)

corpus = unique_corpus
log.info("Removed %d duplicates", initial_count - len(corpus))

# Sort by priority (higher priority first)
corpus.sort(key=lambda x: x.get('priority', 999))

# === FALLBACK: Ensure we always have some data ===
if len(corpus) == 0:
    log.warning("No datasets found - creating fallback entries")
    corpus = [
        {
            'id': 'fallback_1', 'type': 'fallback', 'priority': 999,
            'text': 'symptoms: fever cough headache fatigue', 
            'diagnosis': 'Viral Infection'
        },
        {
            'id': 'fallback_2', 'type': 'fallback', 'priority': 999,
            'text': 'symptoms: sore throat runny nose sneezing', 
            'diagnosis': 'Common Cold'
        },
        {
            'id': 'fallback_3', 'type': 'fallback', 'priority': 999,
            'text': 'symptoms: chest pain shortness of breath', 
            'diagnosis': 'Respiratory Issue'
        },
    ]

# === FINAL CORPUS STATISTICS ===
type_counts = {}
for item in corpus:
    t = item['type']
    type_counts[t] = type_counts.get(t, 0) + 1

corpus_composition = ', '.join([f"{k}: {v}" for k, v in type_counts.items()])
log.info("FINAL CORPUS BUILT: %d total entries (Sources: %s)", len(corpus), corpus_composition)

# Create DataFrame and prepare for TF-IDF
corpus_df = pd.DataFrame(corpus)
corpus_df['clean_text'] = corpus_df['text'].apply(clean)

log.info("Training TF-IDF model on %d unique entries...", len(corpus_df))
try:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=2)
    tfidf_matrix = vectorizer.fit_transform(corpus_df['clean_text'])
    log.info("✓ TF-IDF vocabulary size: %d terms", len(vectorizer.vocabulary_))
except Exception as e:
    log.error("TF-IDF failed: %s", e)
    vectorizer = tfidf_matrix = None

# ---------- MATCH (SYMPTOMS-BASED) ----------
def match(sym, top_k=5):
    if not sym or tfidf_matrix is None:
        return []
    try:
        s = clean(sym)
        v = vectorizer.transform([s])
        sims = cosine_similarity(v, tfidf_matrix).flatten()
        score_by_diag = defaultdict(float)
        count_by_diag = defaultdict(int)
        for idx, sim in enumerate(sims):
            diag = corpus_df.loc[idx, 'diagnosis'] or 'Unknown'
            score_by_diag[diag] += float(sim)
            count_by_diag[diag] += 1
        results = []
        for diag, score in score_by_diag.items():
            avg_score = score / max(1, count_by_diag[diag])
            results.append((diag, avg_score))
        results.sort(key=lambda x: x[1], reverse=True)
        # filter low-scoring matches - keep > small threshold
        filtered = [r for r in results if r[1] > 0.01]
        return filtered[:top_k]
    except Exception as e:
        log.error("Match error: %s", e)
        return [("Error", 0.0)]

# ---------- IMPROVED LOOKUP FUNCTIONS ----------
def lookup_table(df, key_col, value_col, diagnosis):
    if df is None or diagnosis is None:
        return []
    
    diag_l = diagnosis.strip().lower()
    results = []
    
    # Try exact match first
    if key_col in df.columns:
        exact_matches = df[df[key_col].astype(str).str.lower() == diag_l]
        if len(exact_matches) > 0:
            results.extend(exact_matches[value_col].dropna().astype(str).tolist())
    
    # Try partial matches if no exact matches found
    if not results and key_col in df.columns:
        # Split diagnosis into words and try to match any part
        diag_words = diag_l.split()
        for word in diag_words:
            if len(word) > 3:  # Only search for words longer than 3 characters
                contains_matches = df[df[key_col].astype(str).str.lower().str.contains(word, na=False)]
                if len(contains_matches) > 0:
                    results.extend(contains_matches[value_col].dropna().astype(str).tolist())
    
    # Fallback: search any column for the diagnosis
    if not results:
        for c in df.columns:
            if c != value_col:  # Don't search the value column itself
                subset = df[df[c].astype(str).str.lower().str.contains(diag_l, na=False)]
                if len(subset) > 0:
                    results.extend(subset[value_col].dropna().astype(str).tolist())
                    break
    
    # Return unique results
    return list(dict.fromkeys(results))

def meds(d):
    df = datasets.get('medicine')
    if df is None:
        log.warning("Medicine dataset not available")
        return []
    
    # More flexible column detection
    key_col = next((c for c in df.columns if any(x in c.lower() for x in ['disease', 'condition', 'diagnosis', 'illness', 'disorder'])), df.columns[0])
    val_col = next((c for c in df.columns if any(x in c.lower() for x in ['medicine', 'medication', 'drug', 'treatment', 'prescription', 'therapy'])), None)
    
    if val_col is None:
        val_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
    
    log.info("Medicine lookup: diagnosis='%s', key_col='%s', val_col='%s'", d, key_col, val_col)
    results = lookup_table(df, key_col, val_col, d)
    log.info("Found medicine results: %s", results)
    return results[:5]

def herbs(d):
    df = datasets.get('herbal')
    if df is None:
        log.warning("Herbal dataset not available")
        return []
    
    key_col = next((c for c in df.columns if any(x in c.lower() for x in ['disease', 'condition', 'diagnosis'])), df.columns[0])
    val_col = next((c for c in df.columns if any(x in c.lower() for x in ['herbal', 'remedy', 'treatment', 'plant', 'natural'])), None)
    
    if val_col is None:
        val_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
    
    log.info("Herbal lookup: diagnosis='%s', key_col='%s', val_col='%s'", d, key_col, val_col)
    results = lookup_table(df, key_col, val_col, d)
    log.info("Found herbal results: %s", results)
    return results[:5]

def nutrition(d):
    df = datasets.get('nutrition')
    if df is None:
        log.warning("Nutrition dataset not available")
        return []
    
    key_col = next((c for c in df.columns if any(x in c.lower() for x in ['disease', 'condition', 'diagnosis'])), df.columns[0])
    val_col = next((c for c in df.columns if any(x in c.lower() for x in ['nutrition', 'food', 'diet', 'supplement', 'vitamin'])), None)
    
    if val_col is None:
        val_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
    
    log.info("Nutrition lookup: diagnosis='%s', key_col='%s', val_col='%s'", d, key_col, val_col)
    results = lookup_table(df, key_col, val_col, d)
    log.info("Found nutrition results: %s", results)
    return results[:5]

# ---------- IMPROVED LIFESTYLE RECOMMENDATIONS ----------
def extract_lifestyle_from_careplan(careplan_text):
    lifestyle_terms = []
    text_lower = careplan_text.lower()
    
    if 'diet' in text_lower:
        if 'modification' in text_lower:
            lifestyle_terms.append('Follow dietary modifications as prescribed')
        else:
            lifestyle_terms.append('Maintain healthy diet')
    
    if 'exercise' in text_lower or 'physical activity' in text_lower:
        lifestyle_terms.append('Regular physical activity as recommended')
    
    if 'rehabilitation' in text_lower:
        lifestyle_terms.append('Participate in recommended rehabilitation program')
    
    if 'therapy' in text_lower and 'behavioral' in text_lower:
        lifestyle_terms.append('Engage in behavioral therapy sessions')
    
    if 'monitoring' in text_lower:
        lifestyle_terms.append('Regular self-monitoring as advised')
    
    return lifestyle_terms if lifestyle_terms else ['Follow prescribed care plan']

def get_generic_lifestyle_recommendations(diagnosis):
    diag = (diagnosis or '').lower()
    recs = []
    
    if any(x in diag for x in ['migraine', 'headache']):
        recs += [
            'Identify and avoid headache triggers',
            'Maintain regular sleep schedule',
            'Stay hydrated',
            'Manage stress through relaxation techniques'
        ]
    elif 'hypertension' in diag or 'high blood pressure' in diag:
        recs += [
            'Reduce salt intake',
            'Maintain healthy weight',
            'Limit alcohol consumption',
            'Regular aerobic exercise'
        ]
    elif 'diabetes' in diag:
        recs += [
            'Monitor blood glucose regularly',
            'Reduce refined carbohydrates and sugary drinks',
            'Increase physical activity'
        ]
    elif 'asthma' in diag:
        recs += [
            'Avoid known triggers (smoke, pollen, dust)',
            'Follow inhaler/medication plan',
            'Consider allergen-proofing home'
        ]
    elif 'heart' in diag:
        recs += [
            'Cardiac-friendly diet low in saturated fats',
            'Regular moderate exercise',
            'Stress management techniques'
        ]
    elif 'kidney' in diag:
        recs += [
            'Follow renal diet restrictions',
            'Monitor fluid intake',
            'Regular kidney function tests'
        ]
    elif 'obesity' in diag:
        recs += [
            'Calorie-controlled balanced diet',
            'Regular physical activity',
            'Behavior modification for eating habits'
        ]
    elif 'depression' in diag:
        recs += [
            'Regular sleep schedule',
            'Social engagement and support',
            'Mindfulness and relaxation techniques'
        ]
    elif 'infection' in diag or 'fever' in diag:
        recs += [
            'Stay hydrated',
            'Rest and seek medical evaluation if symptoms worsen'
        ]
    
    general_recs = ['Maintain balanced diet', 'Regular exercise', 'Get routine checkups']
    all_recs = recs + general_recs
    return list(dict.fromkeys(all_recs))[:5]

def lifestyle(d):
    """Get lifestyle recommendations from careplans dataset first, then fallback to generic"""
    if 'careplans' not in datasets:
        return get_generic_lifestyle_recommendations(d)
    
    cp_df = datasets['careplans']
    diag_col = next((c for c in cp_df.columns if 'diag' in c.lower()), cp_df.columns[0])
    plan_col = next((c for c in cp_df.columns if 'care' in c.lower() or 'plan' in c.lower()), None)
    
    if plan_col is None:
        return get_generic_lifestyle_recommendations(d)
    
    diag_l = (d or '').strip().lower()
    
    # Try exact match first
    exact_matches = cp_df[cp_df[diag_col].astype(str).str.lower() == diag_l]
    if len(exact_matches) > 0:
        plans = exact_matches[plan_col].dropna().astype(str).tolist()
        if plans:
            lifestyle_advice = []
            for plan in plans:
                lifestyle_advice.extend(extract_lifestyle_from_careplan(plan))
            
            if lifestyle_advice:
                return list(dict.fromkeys(lifestyle_advice))[:5]
    
    # Try partial match
    for word in diag_l.split():
        if len(word) > 4:
            partial_matches = cp_df[cp_df[diag_col].astype(str).str.lower().str.contains(word, na=False)]
            if len(partial_matches) > 0:
                plans = partial_matches[plan_col].dropna().astype(str).tolist()
                if plans:
                    lifestyle_advice = []
                    for plan in plans:
                        lifestyle_advice.extend(extract_lifestyle_from_careplan(plan))
                    
                    if lifestyle_advice:
                        return list(dict.fromkeys(lifestyle_advice))[:5]
    
    return get_generic_lifestyle_recommendations(d)

# ---------- COMPOSE (BEAUTIFUL FORMAT) ----------
def compose(diag, index):
    # Different titles for each position
    titles = ["🩺 Most Likely Cause", "🔍 Also Consider", "💡 Could Also Be"]
    title = titles[index] if index < len(titles) else f"Possible Cause #{index+1}"
    
    lines = [f"<div class='diagnosis-title'>{title}: <span class='condition-name'>{diag}</span></div>"]
    
    # Medicines with pill icon
    m = meds(diag)
    if m:
        med_list = " • ".join(m)
        lines.append(f"<div class='recommendation-section'><div class='section-title'>💊 Recommended Medicines</div><div class='item-list'>{med_list}</div></div>")
    else:
        lines.append(f"<div class='recommendation-section'><div class='section-title'>💊 Recommended Medicines</div><div class='item-list no-data'>No specific medicine recommendations found</div></div>")
    
    # Herbal treatments with leaf icon
    h = herbs(diag)
    if h:
        herb_list = " • ".join(h)
        lines.append(f"<div class='recommendation-section'><div class='section-title'>🌿 Herbal Treatments</div><div class='item-list'>{herb_list}</div></div>")
    else:
        lines.append(f"<div class='recommendation-section'><div class='section-title'>🌿 Herbal Treatments</div><div class='item-list no-data'>No herbal recommendations found</div></div>")
    
    # Nutrition with apple icon
    n = nutrition(diag)
    if n:
        nutrition_list = " • ".join(n)
        lines.append(f"<div class='recommendation-section'><div class='section-title'>🍎 Nutrition Advice</div><div class='item-list'>{nutrition_list}</div></div>")
    else:
        lines.append(f"<div class='recommendation-section'><div class='section-title'>🍎 Nutrition Advice</div><div class='item-list no-data'>No specific nutrition advice found</div></div>")
    
    # Lifestyle with heart icon
    l = lifestyle(diag)
    if l:
        lifestyle_list = " • ".join(l)
        lines.append(f"<div class='recommendation-section'><div class='section-title'>❤️ Lifestyle Recommendations</div><div class='item-list'>{lifestyle_list}</div></div>")
    
    return "<div class='diagnosis-container'>" + "".join(lines) + "</div>"

# ---------- RUN (COMPREHENSIVE CORPUS) ----------
def run(sym, age="", gender=""):
    ctx_parts = [sym]
    if age: ctx_parts.append("age " + age)
    if gender: ctx_parts.append("gender " + gender)
    ctx = " ".join(ctx_parts)
    
    matches = match(ctx, top_k=3)
    
    if not matches:
        return "<div class='no-results'>No matching conditions found for your symptoms. Please consult a healthcare professional for proper diagnosis.</div>"
    
    # Header with patient info
    header_parts = ["<div class='assessment-header'>"]
    header_parts.append("<div class='main-title'>Health Assessment Results</div>")
    header_parts.append(f"<div class='symptoms'>Based on your symptoms: <span class='symptom-text'>{sym}</span></div>")
    
    if age or gender:
        info_parts = []
        if age: info_parts.append(f"{age} years old")
        if gender: info_parts.append(gender)
        header_parts.append(f"<div class='patient-info'>Patient: {', '.join(info_parts)}</div>")
    
    header_parts.append("<div class='recommendations-title'>Recommended Approaches:</div>")
    header_parts.append("</div>")
    
    # Compose recommendations
    parts = [compose(d, i) for i, (d, s) in enumerate(matches[:3])]
    
    # Footer with continuation option
    footer = """
    <div class='footer'>
        <div class='footer-note'>💡 <b>Remember:</b> This is for informational purposes only. Always consult a healthcare professional for proper diagnosis and treatment.</div>
        <div class='continue-prompt'>You can describe more symptoms for another assessment, or ask questions about any recommendation.</div>
    </div>
    """
    
    return "".join(header_parts) + "<div class='diagnoses-wrapper'>" + "".join(parts) + "</div>" + footer

# ------------------- FLASK -------------------
app = Flask(__name__, static_folder=DATA_DIR)

@app.route("/")
def index():
    p = os.path.join(DATA_DIR, "index.html")
    return send_from_directory(DATA_DIR, "index.html") if os.path.exists(p) else "index.html missing"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        sym = data.get("message", "").strip()
        age = data.get("age", "")
        gen = data.get("gender", "")
        if not sym:
            return jsonify({"english": "Please describe your symptoms."})
        out = run(sym, age, gen)
        return jsonify({"english": out})
    except Exception as e:
        log.error("PREDICT CRASH: %s", e, exc_info=True)
        return jsonify({"english": "Sorry, something went wrong. Please try again."}), 500

@app.route("/health")
def health():
    # Calculate corpus composition for health endpoint
    type_counts = {}
    for item in corpus:
        t = item['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    primary_source = "symptoms.csv" if type_counts.get('symptoms', 0) > 0 else "observations/conditions"
    
    return jsonify({
        "status": "ok",
        "corpus": len(corpus),
        "corpus_composition": type_counts,
        "primary_source": primary_source,
        "tfidf": tfidf_matrix is not None,
        "datasets_loaded": list(datasets.keys())
    })

# ignore favicon 404 – harmless
@app.route("/favicon.ico")
def favicon(): return "", 204

if __name__ == "__main__":
    log.info("Starting Health Advisor (Comprehensive Corpus) on %s:%d", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False)

