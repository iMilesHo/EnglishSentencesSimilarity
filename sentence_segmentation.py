import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.data import find
import ssl

# avoid SSL certificate error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# check if punkt tokenizer models are downloaded
try:
    find('tokenizers/punkt')
    print("Punkt Tokenizer Models are already downloaded.")
except LookupError:
    print("Punkt Tokenizer Models not found. Downloading them...")
    nltk.download('punkt')
try:
    find('taggers/averaged_perceptron_tagger')
    print("averaged_perceptron_tagger are already downloaded.")
except LookupError:
    print("averaged_perceptron_tagger not found. Downloading them...")
    nltk.download('averaged_perceptron_tagger')

def tokenize_into_sentences(text):
    return nltk.sent_tokenize(text)

def align_sentences_enhanced(original_sentences, recognized_sentences, tolerance=0.36, max_merge=30):
    original_sentences = tokenize_into_sentences(original_sentences)
    recognized_sentences = tokenize_into_sentences(recognized_sentences)

    aligned_sentences = []
    i, j = 0, 0

    while i < len(original_sentences) and j < len(recognized_sentences):
        # Get lengths of the current sentences
        original_len = len(original_sentences[i].split())
        recognized_len = len(recognized_sentences[j].split())

        # Function to check and merge next few sentences
        def check_and_merge_sentences(target_len, sentences, index, max_merge):
            combined_sentence = sentences[index]
            current_len = len(combined_sentence.split())
            for k in range(1, max_merge):
                if index + k < len(sentences):
                    combined_sentence += " " + sentences[index + k]
                    new_len = len(combined_sentence.split())
                    if abs(new_len - target_len) / max(new_len, target_len) < tolerance:
                        return combined_sentence, k + 1
                else:
                    break
            return combined_sentence, 1

        if abs(original_len - recognized_len) / max(original_len, recognized_len) < tolerance:
            aligned_sentences.append((original_sentences[i], recognized_sentences[j]))
            i += 1
            j += 1
        elif original_len < recognized_len:
            merged_sentence, k = check_and_merge_sentences(recognized_len, original_sentences, i, max_merge)
            aligned_sentences.append((merged_sentence, recognized_sentences[j]))
            i += k
            j += 1
        else:
            merged_sentence, k = check_and_merge_sentences(original_len, recognized_sentences, j, max_merge)
            aligned_sentences.append((original_sentences[i], merged_sentence))
            i += 1
            j += k

    # Check and handle the last aligned pair
    if aligned_sentences:
        last_aligned = aligned_sentences[-1]
        original_last_len = len(last_aligned[0].split())
        recognized_last_len = len(last_aligned[1].split())

        if abs(original_last_len - recognized_last_len) / max(original_last_len, recognized_last_len) > tolerance:
            # Determine which sentence is longer and trim it
            if original_last_len > recognized_last_len:
                trimmed_sentence = ' '.join(last_aligned[0].split()[:round(recognized_last_len*1.49)])
                aligned_sentences[-1] = (trimmed_sentence, last_aligned[1])
            elif recognized_last_len > original_last_len:
                trimmed_sentence = ' '.join(last_aligned[1].split()[:round(original_last_len*1.49)])
                aligned_sentences[-1] = (last_aligned[0], trimmed_sentence)

    return aligned_sentences

if __name__ == '__main__':
    # Example usage
    original_text = """
    The Great Wall of China, a marvel of engineering stretching over 13,000 miles, is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was constructed over several centuries, beginning as early as the 7th century BC, with the most renowned portions built during the Ming Dynasty (1368–1644 AD). Initially erected by various states to protect against northern invasions, the Wall was later unified and expanded to defend the Chinese Empire against nomadic tribes. Its winding path over rugged country and steep mountains showcases the ancient world's immense determination and resourcefulness. Today, the Great Wall stands as a UNESCO World Heritage site and a symbol of China’s historical resilience, although it faces challenges such as erosion and damage from tourism and development.
    """
    recognized_text = """
    The great all of China I\'m over on engineering student University Inn selves and Miles is a service of fortifications made of Stone by tens are good and other materials. It was construct it over Sarah centuries beginning as early as the 17th century BC. Where is the most Reno and portions build to join the main dynasty initially directed by marriage. Stays to protect against North dimensions."""
    aligned = align_sentences_enhanced(original_text, recognized_text)
    for orig, recog in aligned:
        print(f"Original:\n {orig}\nRecognized:\n {recog}\n")

