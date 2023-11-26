import logging
from Siamese_Model import predict
from sentence_segmentation import align_sentences_enhanced

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_sentences(sentence1, sentence2, threshold=0.7):
    """
    Compare two sentences using the Siamese model.
    """
    return predict(sentence1, sentence2, threshold, verbose=True)

def main():
    # Examples of comparing questions
    examples = [
        ("When will I see you?", "When can I see you again?"),
        ("How old are you?", "What is your age?")
    ]

    for q1, q2 in examples:
        logger.info(f"Comparing: '{q1}' with '{q2}'")
        compare_sentences(q1, q2)

    # Examples of comparing speeches
    speeches = [
        ("""Halifax is a historic port city and the capital of the Canadian province of Nova Scotia. Nestled on the country's east coast, Halifax is known for its rich maritime heritage and vibrant cultural scene. The city is home to the iconic Halifax Citadel, a star-shaped fortress that offers panoramic views of the harbor. Boasting a diverse population, Halifax is celebrated for its friendly locals and welcoming atmosphere. Visitors can explore its charming waterfront, enjoy fresh seafood, and immerse themselves in the city's lively arts and music scene""",
         """Halifax is a historic port city located in the Canadian province of Nova Scotia. Nestled on the eastern coast of the country, Halifax is renowned for its rich maritime heritage, dating back to its founding in 1749. The city is home to the iconic Citadel Hill, a National Historic Site that offers panoramic views of the harbor and serves as a reminder of Halifax's military past. Boasting a vibrant cultural scene, Halifax hosts numerous festivals, museums, and galleries that celebrate its diverse history and artistic achievements. With its friendly locals, picturesque waterfront, and a thriving culinary scene, Halifax offers a welcoming atmosphere that attracts both residents and visitors alike"""),
        (
        """Halifax is a historic port city and the capital of the Canadian province of Nova Scotia. Nestled on the country's east coast, Halifax is known for its rich maritime heritage and vibrant cultural scene. The city is home to the iconic Halifax Citadel, a star-shaped fortress that offers panoramic views of the harbor. Boasting a diverse population, Halifax is celebrated for its friendly locals and welcoming atmosphere. Visitors can explore its charming waterfront, enjoy fresh seafood, and immerse themselves in the city's lively arts and music scene""",
        """Toronto, the largest city in Canada, is a dynamic and cosmopolitan metropolis situated in the province of Ontario. Known for its striking skyline dominated by the iconic CN Tower, Toronto is a global financial and cultural hub. The city is celebrated for its cultural diversity, reflected in vibrant neighborhoods like Kensington Market and Chinatown, where various ethnic communities thrive. Toronto is also home to world-class attractions such as the Royal Ontario Museum and the Art Gallery of Ontario, contributing to its reputation as a cultural powerhouse. With a bustling downtown core, diverse culinary offerings, and a robust public transportation system, Toronto exemplifies a modern, inclusive urban experience. """),
        ("""The Great Wall of China, a marvel of engineering stretching over 13,000 miles, is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was constructed over several centuries, beginning as early as the 7th century BC, with the most renowned portions built during the Ming Dynasty (1368–1644 AD). Initially erected by various states to protect against northern invasions, the Wall was later unified and expanded to defend the Chinese Empire against nomadic tribes. Its winding path over rugged country and steep mountains showcases the ancient world's immense determination and resourcefulness. Today, the Great Wall stands as a UNESCO World Heritage site and a symbol of China’s historical resilience, although it faces challenges such as erosion and damage from tourism and development.""",
         """The great all of China I\'m over on engineering student University Inn selves and Miles is a service of fortifications made of Stone by tens are good and other materials. It was construct it over Sarah centuries beginning as early as the 17th century BC. Where is the most Reno and portions build to join the main dynasty initially directed by marriage. Stays to protect against North dimensions.""")
    ]

    for s1, s2 in speeches:
        logger.info("\n------------------------")
        logger.info("Comparing two speeches")
        aligned_sentences = align_sentences_enhanced(s1, s2)
        similarities = [compare_sentences(orig, recog) for orig, recog in aligned_sentences]
        logger.info(f"Similarities: {similarities}")

if __name__ == '__main__':
    main()
