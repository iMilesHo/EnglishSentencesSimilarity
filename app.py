from flask import Flask, request, jsonify
from Siamese_Model import predict as get_similarity_score
from sentence_segmentation import align_sentences_enhanced
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a file handler for logging
file_handler = RotatingFileHandler('./log/app.log', maxBytes=10000, backupCount=1)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
# Add the file handler to the logger
logger.addHandler(file_handler)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    original_sentences = data['original_sentences']
    recognized_sentences = data['recognized_sentences']
    logger.info(f"Received request with original and recognized sentences.")
    logger.info(f"original sentences:\n {original_sentences}")
    logger.info(f"recognized sentences:\n {recognized_sentences}")

    # Assume align_sentences_enhanced and other necessary functions are defined
    aligned_sentences = align_sentences_enhanced(original_sentences, recognized_sentences)

    results = []
    for original, recognized in aligned_sentences:
        # Convert sentences to model input format, get score
        res, score = get_similarity_score(original, recognized, 0.7, verbose=True)
        results.append({'original': original, 'recognized': recognized, 'score': score})
    logger.info(f"Results: \n{results}")
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Runs the server
