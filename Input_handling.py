import re
import nltk
import json
import operator
from deep_translator import GoogleTranslator
from langdetect import detect
from nltk.corpus import stopwords
from difflib import SequenceMatcher

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------
def load_json_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

CRITICAL_SYMPTOMS = load_json_file('critical_symptoms.json')
NORMAL_SYMPTOMS = load_json_file('normal_symptoms.json')
DEPT_MAP = load_json_file('critical_dept_map.json') 
DOCTOR_MAP = load_json_file('doctor_map.json')
DISEASE_MAP = load_json_file('disease_map.json')

# -----------------------------------------------------------
# 2. UTILS
# -----------------------------------------------------------
def detect_language(text: str) -> str:
    try: return detect(text)
    except: return "en"

def translate_to_english(text: str, lang: str) -> str:
    if lang == "en": return text
    try: return GoogleTranslator(source='auto', target='en').translate(text)
    except: return text

def extract_symptoms(text: str):
    text = text.lower()
    # Normalize punctuation to handle "pain,vomiting" without spaces
    text = re.sub(r'[,.]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Split by logical separators
    parts = re.split(r'\band\b|\bwith\b|\bdue to\b|\bbecause of\b', text)
    return [p.strip() for p in parts if p.strip()]

# -----------------------------------------------------------
# 3. SMART MATCHING LOGIC (THE FIX)
# -----------------------------------------------------------

def is_critical_match(symptom_phrase):
    """
    Returns True if the symptom phrase matches any critical definition
    using Token-Based Matching (Order independent).
    """
    symptom_words = set(symptom_phrase.split())
    
    # 1. Check Explicit JSON Critical List
    for crit in CRITICAL_SYMPTOMS:
        crit_words = set(crit.split())
        # If all words in a critical phrase (e.g. {"vomiting", "blood"}) 
        # represent a subset of the symptom words (e.g. {"blood", "vomiting", "severe"})
        if crit_words.issubset(symptom_words):
            return True
            
    # 2. Check "Hard Rules" (Combinations that are always critical)
    # Heart + Pain
    if "heart" in symptom_words and ("pain" in symptom_words or "attack" in symptom_words):
        return True
    
    # Blood + Output (Vomit/Stool/Cough)
    if "blood" in symptom_words and any(x in symptom_words for x in ["vomit", "vomiting", "stool", "urine", "cough", "coughing"]):
        return True
        
    # Trauma Keywords
    trauma_triggers = ["accident", "crash", "fall", "unconscious", "collapse", "suicide", "poison", "stroke","accident", "crash", "collision", "fall", "fell", "hit by", "injury", 
        "trauma", "wound", "burn", "cut", "bleed", "fracture", "broken bone",
        "electrocution", "drown", "poison", "bite","break", "laceration",
    "contusion",
    "bruise",
    "sprain",
    "dislocation",
    "amputation",
    "concussion",
    "hemorrhage",
    "puncture",
    "crush injury",
    "frostbite",
    "hypothermia",
    "heatstroke",
    "dehydration",
    "suffocation",
    "asphyxiation"]
    if any(t in symptom_phrase for t in trauma_triggers):
        return True

    return False

def calculate_department_score(symptoms, is_critical_flag):
    scores = {dept: 0 for dept in DEPT_MAP.keys()}
    scores["General"] = 0
    
    for s in symptoms:
        s_words = set(s.split())
        
        for dept, keywords in DEPT_MAP.items():
            for key in keywords:
                # Direct word match (High score)
                if key in s_words: 
                    scores[dept] += 5
                # Phrase match (e.g. "chest pain" in "severe chest pain")
                elif key in s:
                    scores[dept] += 5
    
    # PRIORITY OVERRIDES
    # If Critical + Heart involved -> Boost Cardiology
    if is_critical_flag and any("heart" in s or "chest" in s for s in symptoms):
        scores["Cardiology"] += 50

    # If Critical + Accident involved -> Boost Emergency
    if is_critical_flag and any(x in str(symptoms) for x in ["accident", "fall", "crash", "trauma"]):
        scores["Emergency"] += 50
        
    return scores

def determine_status_and_dept(symptoms):
    # 1. Determine Status (Critical vs Normal)
    is_critical = False
    
    for s in symptoms:
        if is_critical_match(s):
            is_critical = True
            break
    
    # 2. Score Departments
    dept_scores = calculate_department_score(symptoms, is_critical)
    
    # 3. Find Winner
    top_dept = max(dept_scores.items(), key=operator.itemgetter(1))[0]
    
    # 4. Final Safety Routing
    if is_critical:
        # If it's critical but mapped to General or something mild, force logic
        if top_dept == "General" or dept_scores[top_dept] < 5:
            # Fallback for unidentified critical issues
            top_dept = "Emergency"
            
    return {
        "status": "Critical" if is_critical else "Normal",
        "department": top_dept
    }

# -----------------------------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------------------------
def medical_pipeline(user_text: str):
    lang = detect_language(user_text)
    english_text = translate_to_english(user_text, lang)
    symptoms = extract_symptoms(english_text)

    analysis = determine_status_and_dept(symptoms)
    
    final_dept = analysis["department"]
    final_status = analysis["status"]
    
    # Get Disease Prediction
    doctor = DOCTOR_MAP.get(final_dept, "General Physician")
    
    # Simple prediction lookup
    prediction = "Medical Condition"
    if final_dept in DISEASE_MAP:
        candidates = []
        for s in symptoms:
            for k, v in DISEASE_MAP[final_dept].items():
                if k in s: candidates.append(v)
        if candidates: prediction = ", ".join(list(set(candidates)))
        elif final_dept == "Emergency": prediction = "Critical Emergency"
        elif final_dept == "Cardiology": prediction = "Cardiac Event"

    # --- CHANGED RETURN STRUCTURE TO MATCH REACT ---
    return {
        "input": user_text,
        "translated_text": english_text,
        "final_status": final_status,
        "disease_info": {  # <--- Nested object created here
            "top_department": final_dept,
            "disease_prediction": prediction,
            "doctor": doctor
        }
    }


