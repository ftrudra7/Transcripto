from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianTokenizer, MarianMTModel, BartTokenizer, BartForConditionalGeneration
from langdetect import detect
import torch

app = Flask(__name__)
CORS(app)

device = "cpu"

trans_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
trans_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en").to(device)

sum_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
sum_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

@app.post("/process-text")
def process_text():
    text = request.json["text"]

    # ✅ Detect language
    detected_lang = detect(text)

    # ✅ Translate only if not English
    if detected_lang != "en":
        batch = trans_tokenizer(text, return_tensors="pt").to(device)
        translated = trans_model.generate(**batch)
        translated_text = trans_tokenizer.decode(translated[0], skip_special_tokens=True)
    else:
        translated_text = text

    # ✅ Summarize
    inputs = sum_tokenizer([translated_text], max_length=1024, return_tensors="pt").to(device)
    summary_ids = sum_model.generate(inputs["input_ids"], max_length=100, min_length=20)
    summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({
        "language_detected": detected_lang,
        "translated": translated_text,
        "summary": summary
    })

if __name__ == "__main__":
    app.run(debug=True)
