import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess

# Load FAQs
with open("data/faqs.json", "r") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# Preprocess
preprocessed_questions = [preprocess(q) for q in questions]

# Vectorize
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(preprocessed_questions)

def get_response(user_query):
    user_query_prep = preprocess(user_query)
    user_vec = vectorizer.transform([user_query_prep])
    
    similarities = cosine_similarity(user_vec, question_vectors)
    idx = np.argmax(similarities)
    
    if similarities[0][idx] < 0.3:  # threshold
        return "Sorry, I don’t understand your question."
    return answers[idx]

# CLI Chatbot
if __name__ == "__main__":
    print("🤖 Chatbot is ready! (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        print("Bot:", get_response(query))
