from flask import Flask, request, jsonify
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from flask_cors import CORS
from PIL import Image
import io
import easyocr

app = Flask(__name__)
reader = easyocr.Reader(['en'])
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load models and tokenizers for both translation options
try:
    model_en_ny = "Helsinki-NLP/opus-mt-en-ny"
    tokenizer_en_ny = AutoTokenizer.from_pretrained(model_en_ny)
    model_en_ny = TFAutoModelForSeq2SeqLM.from_pretrained(model_en_ny)

    model_ny_en = "Helsinki-NLP/opus-mt-ny-en"
    tokenizer_ny_en = AutoTokenizer.from_pretrained(model_ny_en)
    model_ny_en = TFAutoModelForSeq2SeqLM.from_pretrained(model_ny_en)
except Exception as e:
    print("Error loading models:", e)
    exit(1)

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()

    # Check if the required fields are present in the request
    if 'text' not in data or 'direction' not in data:
        return jsonify({'error': 'Missing text or direction in request'}), 400

    text = data['text']
    direction = data['direction']

    # Determine which model and tokenizer to use based on the direction
    if direction == "en-ny":
        tokenizer = tokenizer_en_ny
        model = model_en_ny
    elif direction == "ny-en":
        tokenizer = tokenizer_ny_en
        model = model_ny_en
    else:
        return jsonify({'error': 'Invalid direction specified'}), 400

    try:
        # Perform the translation
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
        outputs = model.generate(inputs["input_ids"])
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'translated_text': translated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    # Perform OCR
    results = reader.readtext(image_bytes)

    # Extract text from results
    extracted_text = ' '.join([text for _, text, _ in results])

    return jsonify({'extracted_text': extracted_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)