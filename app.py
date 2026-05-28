"""
Franco-Arabic ABSA API Server
============================
Run with: uvicorn app:app --reload --port 5000

Model expected at: franco_weights_only.pt
"""

import os
import re
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Franco-Arabic ABSA API")

# CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= MODELS AND CLASSES =============

class FrancoUltraTranslator:
    """Franco-Arabic to Arabic translator"""
    
    def __init__(self):
        self.number_map_primary = {
            '2': 'ا', '3': 'ع', '7': 'ح', '8': 'غ',
            '9': 'ص', '6': 'ط', '5': 'خ', '4': 'ش',
            "'": 'ء',
        }
        self.combinations = {
            'sh': 'ش', 'ch': 'تش', 'th': 'ث', 'kh': 'خ',
            'gh': 'غ', 'dh': 'ذ', 'zh': 'ظ',
            'aa': 'ا', 'ee': 'ي', 'oo': 'و', 'ii': 'ي',
        }
        self.letter_map = {
            'a': 'ا', 'b': 'ب', 't': 'ت', 'j': 'ج', 'h': 'ح',
            'd': 'د', 'r': 'ر', 'z': 'ز', 's': 'س', 'f': 'ف',
            'q': 'ق', 'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن',
            'w': 'و', 'y': 'ي', 'e': 'ع', 'g': 'ج',
        }
        self.arabic_normalization = {
            'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ى': 'ا', 'ة': 'ه',
        }
        self.dictionary = {
            'ana': 'انا', 'enta': 'انت', 'kwayes': 'كويس',
            'mish': 'مش', '3ayez': 'عايز', '7elw': 'حلو',
            'shukran': 'شكرا', 'khalas': 'خلاص', 'tamam': 'تمام',
            'service': 'خدمة', 'food': 'اكل', 'price': 'سعر',
        }

    def transliterate(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        text = text.lower()
        for franco, arabic in self.dictionary.items():
            text = text.replace(franco, arabic)
        for num, arabic in self.number_map_primary.items():
            text = text.replace(num, arabic)
        for combo, arabic in self.combinations.items():
            text = text.replace(combo, arabic)
        words = text.split()
        converted_words = []
        for word in words:
            if any(c.isalpha() and c.isascii() for c in word) and len(word) < 15:
                converted = ''.join(self.letter_map.get(c, c) for c in word)
                converted_words.append(converted)
            else:
                converted_words.append(word)
        text = ' '.join(converted_words)
        for before, after in self.arabic_normalization.items():
            text = text.replace(before, after)
        return text.strip()


class EgyptianSlangHandler:
    def __init__(self):
        self.slang_map = {
            'تحفة': 'ممتاز', 'جامد': 'قوي', 'عظمة': 'رائع',
        }

    def process(self, text):
        if not isinstance(text, str):
            return text
        for slang, std in self.slang_map.items():
            text = text.replace(slang, std)
        return text


class UltimateTextPreprocessor:
    def __init__(self):
        self.franco = FrancoUltraTranslator()
        self.slang = EgyptianSlangHandler()
        self.franco_numbers = ['2', '3', '7', '8', '9', '6', '5', '4']
        self.franco_words = ['ana', 'enta', 'kwayes', 'mish', '3ayez', '7elw']

    def is_franco(self, text):
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return False
        return any(num in text for num in self.franco_numbers) or \
               any(word in text.lower() for word in self.franco_words)

    def clean_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        
        is_franco_text = self.is_franco(text)
        
        if is_franco_text:
            text = self.franco.transliterate(text)
        
        text = self.slang.process(text)
        
        # Clean special characters
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class FrancoABSAModel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout=0.1, num_aspects=9, num_sentiments=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(hidden_size, num_aspects)
        self.sentiment_head = nn.Linear(hidden_size, num_aspects * num_sentiments)
        self.star_embedding = nn.Linear(1, 64)
        self.star_norm = nn.LayerNorm(64)
        self.fusion_layer = nn.Linear(hidden_size + 64, hidden_size)
        self.fusion_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask, star_rating):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        
        if star_rating.dim() == 1:
            star_rating = star_rating.unsqueeze(1)
        star_features = self.star_norm(self.star_embedding(star_rating))
        
        combined = torch.cat([pooled, star_features], dim=1)
        fused = self.dropout(self.fusion_norm(self.fusion_layer(combined)))
        aspect_logits = self.aspect_head(fused)
        sentiment_logits = self.sentiment_head(fused).view(-1, self.num_aspects, self.num_sentiments)
        return aspect_logits, sentiment_logits


# ============= GLOBALS =============

ASPECTS = ["food", "service", "price", "cleanliness", "delivery", 
           "ambiance", "app_experience", "general", "none"]
SENTIMENTS = ["positive", "negative", "neutral"]

MODEL_PATH = "franco_weights_only.pt"
MODEL_NAME = "xlm-roberta-base"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None
_preprocessor = None
_model_loaded = False


# ============= LOAD MODEL =============

def load_model():
    global _tokenizer, _model, _preprocessor, _model_loaded
    
    if _model_loaded:
        return True
    
    try:
        print(f"Loading model on {_device}...")
        _preprocessor = UltimateTextPreprocessor()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = FrancoABSAModel(MODEL_NAME, dropout=0.1).to(_device)
        
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=_device)
            _model.load_state_dict(state_dict)
            print(f"✅ Loaded model weights from {MODEL_PATH}")
        else:
            print(f"⚠️ Model file not found: {MODEL_PATH}")
            print("   Running in demo mode with random weights")
        
        _model.eval()
        _model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# ============= API MODELS =============

class PredictRequest(BaseModel):
    text: str
    star_rating: float = 3.0
    threshold: float = 0.5

class PredictResponse(BaseModel):
    original: str
    clean: str
    aspects: List[str]
    aspect_sentiments: Dict[str, str]
    aspect_probs: Dict[str, float]
    detected_lang: str
    was_franco: bool
    star_rating: float

class TranslateRequest(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    original: str
    translated: str
    detected_dialect: str


# ============= API ENDPOINTS =============

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/")
async def root():
    return {"message": "Franco-Arabic ABSA API Server", "status": "running"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": _model_loaded}

@app.get("/api/status")
async def status():
    """Get model status - for React frontend"""
    return {
        "loaded": _model_loaded,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "demo_mode": not os.path.exists(MODEL_PATH),
        "device": str(_device)
    }

@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not _model_loaded:
        if not load_model():
            raise HTTPException(status_code=500, detail="Model not loaded")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Preprocess
    clean = _preprocessor.clean_text(text)
    was_franco = _preprocessor.is_franco(text)
    
    if len(clean) < 2:
        return PredictResponse(
            original=text,
            clean=clean,
            aspects=["none"],
            aspect_sentiments={},
            aspect_probs={},
            detected_lang="unknown",
            was_franco=was_franco,
            star_rating=request.star_rating
        )
    
    # Prepare input
    star_norm = (request.star_rating - 1) / 4
    star_tensor = torch.tensor([star_norm], dtype=torch.float).to(_device)
    
    inputs = _tokenizer(clean, return_tensors="pt", truncation=True, 
                        max_length=128, padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        aspect_logits, sentiment_logits = _model(
            inputs["input_ids"], inputs["attention_mask"], star_tensor
        )
        aspect_probs = torch.sigmoid(aspect_logits).cpu().numpy()[0]
        sentiment_preds = sentiment_logits.argmax(-1).cpu().numpy()[0]
    
    # Process results
    detected_aspects = [ASPECTS[i] for i, p in enumerate(aspect_probs) 
                        if p >= request.threshold]
    
    if "none" in detected_aspects and len(detected_aspects) > 1:
        detected_aspects.remove("none")
    if not detected_aspects:
        detected_aspects = ["none"]
    
    aspect_sentiments = {}
    for asp in detected_aspects:
        if asp != "none":
            idx = ASPECTS.index(asp)
            aspect_sentiments[asp] = SENTIMENTS[sentiment_preds[idx]]
    
    return PredictResponse(
        original=text,
        clean=clean,
        aspects=detected_aspects,
        aspect_sentiments=aspect_sentiments,
        aspect_probs={ASPECTS[i]: float(p) for i, p in enumerate(aspect_probs)},
        detected_lang="arabic" if any('\u0600' <= c <= '\u06FF' for c in clean) else "english",
        was_franco=was_franco,
        star_rating=request.star_rating
    )

@app.post("/api/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    translator = FrancoUltraTranslator()
    translated = translator.transliterate(text)
    
    return TranslateResponse(
        original=text,
        translated=translated,
        detected_dialect="egyptian"
    )


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  Franco-Arabic ABSA API Server")
    print("="*50)
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Server: http://localhost:5000")
    print(f"  API docs: http://localhost:5000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=5000)