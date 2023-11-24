from Siamese_Model import predict
from sentence_segmentation import align_sentences_enhanced
import nltk
import torch

# Example usage
question1 = "When will I see you?"
question2 = "When can I see you again?"
predict(question1, question2, 0.7, verbose=True)

question1 = "When will I see you?"
question2 = "When will I see you?"
predict(question1, question2, 0.7, verbose=True)

question1 = "How old are you?"
question2 = "What is your age?"
predict(question1, question2, 0.7, verbose=True)

speechOne ='''Halifax is a historic port city and the capital of the Canadian province of Nova Scotia. Nestled on the country's east coast, Halifax is known for its rich maritime heritage and vibrant cultural scene. The city is home to the iconic Halifax Citadel, a star-shaped fortress that offers panoramic views of the harbor. Boasting a diverse population, Halifax is celebrated for its friendly locals and welcoming atmosphere. Visitors can explore its charming waterfront, enjoy fresh seafood, and immerse themselves in the city's lively arts and music scene'''
print(speechOne)
speechTwo ='''Halifax is a historic port city located in the Canadian province of Nova Scotia. Nestled on the eastern coast of the country, Halifax is renowned for its rich maritime heritage, dating back to its founding in 1749. The city is home to the iconic Citadel Hill, a National Historic Site that offers panoramic views of the harbor and serves as a reminder of Halifax's military past. Boasting a vibrant cultural scene, Halifax hosts numerous festivals, museums, and galleries that celebrate its diverse history and artistic achievements. With its friendly locals, picturesque waterfront, and a thriving culinary scene, Halifax offers a welcoming atmosphere that attracts both residents and visitors alike'''
print(speechTwo)
speechThree = '''
Toronto, the largest city in Canada, is a dynamic and cosmopolitan metropolis situated in the province of Ontario. Known for its striking skyline dominated by the iconic CN Tower, Toronto is a global financial and cultural hub. The city is celebrated for its cultural diversity, reflected in vibrant neighborhoods like Kensington Market and Chinatown, where various ethnic communities thrive. Toronto is also home to world-class attractions such as the Royal Ontario Museum and the Art Gallery of Ontario, contributing to its reputation as a cultural powerhouse. With a bustling downtown core, diverse culinary offerings, and a robust public transportation system, Toronto exemplifies a modern, inclusive urban experience. '''
print(speechThree)

align_sentences_0 = align_sentences_enhanced(speechOne, speechTwo)


# original_text = """
# The Great Wall of China, a marvel of engineering stretching over 13,000 miles, is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was constructed over several centuries, beginning as early as the 7th century BC, with the most renowned portions built during the Ming Dynasty (1368–1644 AD). Initially erected by various states to protect against northern invasions, the Wall was later unified and expanded to defend the Chinese Empire against nomadic tribes. Its winding path over rugged country and steep mountains showcases the ancient world's immense determination and resourcefulness. Today, the Great Wall stands as a UNESCO World Heritage site and a symbol of China’s historical resilience, although it faces challenges such as erosion and damage from tourism and development.
# """
# recognized_text = """
# The great all of China I\'m over on engineering student University Inn selves and Miles is a service of fortifications made of Stone by tens are good and other materials. It was construct it over Sarah centuries beginning as early as the 17th century BC. Where is the most Reno and portions build to join the main dynasty initially directed by marriage. Stays to protect against North dimensions.
# """
#
# align_sentences = align_sentences_enhanced(original_text, recognized_text)

# Compute similarities for each sentence pair
similarities = []
for orig, recog in align_sentences_0:
    similarity = predict(orig, recog, 0.7, verbose=True)
    similarities.append(similarity)

print(similarities)

if __name__ == '__main__':
    pass
