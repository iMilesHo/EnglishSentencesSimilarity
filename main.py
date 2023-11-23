from Siamese_Model import predict
import numpy as np
import pandas as pd
import nltk
import torch
from collections import defaultdict
import pickle

# Load the vocab dictionary from the file
with open('./myResource/vocab.pkl', 'rb') as f:
    loaded_vocab_dict = pickle.load(f)
# Convert it back to a defaultdict if necessary
vocab = defaultdict(lambda: 0, loaded_vocab_dict)

# To load the model later
loaded_model = torch.load('./model/model_entire.pth')
loaded_model.eval()  # Set the model to evaluation mode

# Example usage
question1 = "When will I see you?"
question2 = "When can I see you again?"
predict(question1, question2, 0.7, loaded_model, vocab, verbose=True)

question1 = "When will I see you?"
question2 = "When will I see you?"
predict(question1, question2, 0.7, loaded_model, vocab, verbose=True)

question1 = "How old are you?"
question2 = "What is your age?"
predict(question1, question2, 0.7, loaded_model, vocab, verbose=True)

speechOne ='''Halifax is a historic port city and the capital of the Canadian province of Nova Scotia. Nestled on the country's east coast, Halifax is known for its rich maritime heritage and vibrant cultural scene. The city is home to the iconic Halifax Citadel, a star-shaped fortress that offers panoramic views of the harbor. Boasting a diverse population, Halifax is celebrated for its friendly locals and welcoming atmosphere. Visitors can explore its charming waterfront, enjoy fresh seafood, and immerse themselves in the city's lively arts and music scene'''
print(speechOne)
speechTwo ='''Halifax is a historic port city located in the Canadian province of Nova Scotia. Nestled on the eastern coast of the country, Halifax is renowned for its rich maritime heritage, dating back to its founding in 1749. The city is home to the iconic Citadel Hill, a National Historic Site that offers panoramic views of the harbor and serves as a reminder of Halifax's military past. Boasting a vibrant cultural scene, Halifax hosts numerous festivals, museums, and galleries that celebrate its diverse history and artistic achievements. With its friendly locals, picturesque waterfront, and a thriving culinary scene, Halifax offers a welcoming atmosphere that attracts both residents and visitors alike'''
print(speechTwo)
speechThree = '''
Toronto, the largest city in Canada, is a dynamic and cosmopolitan metropolis situated in the province of Ontario. Known for its striking skyline dominated by the iconic CN Tower, Toronto is a global financial and cultural hub. The city is celebrated for its cultural diversity, reflected in vibrant neighborhoods like Kensington Market and Chinatown, where various ethnic communities thrive. Toronto is also home to world-class attractions such as the Royal Ontario Museum and the Art Gallery of Ontario, contributing to its reputation as a cultural powerhouse. With a bustling downtown core, diverse culinary offerings, and a robust public transportation system, Toronto exemplifies a modern, inclusive urban experience. '''
print(speechThree)


if __name__ == '__main__':
    pass
