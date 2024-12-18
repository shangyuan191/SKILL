# Import the necessary modules
import yaml
from skllm.datasets import get_classification_dataset
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
from skllm.models.gpt.vectorization import GPTVectorizer
from skllm.models.gpt.text2text.translation import GPTTranslator
from skllm.models.gpt.text2text.summarization import GPTSummarizer
from skllm.models.gpt.tagging.ner import GPTExplainableNER
from sklearn.model_selection import train_test_split
# from skllm.datasets import translation
import openai
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'skllm')))
# Load the openai_token from config.yaml
def load_openai_key_from_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get("openai_token")

def load_groq_key_from_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get("groq_token")





if __name__=="__main__":
    # Path to your config.yaml
    config_path = "config.yaml"

    # # Set the OpenAI key
    # openai_token = load_openai_key_from_config(config_path)
    groq_token=load_groq_key_from_config(config_path)
    SKLLMConfig.set_openai_key(groq_token)







    ## Zero-Shot Text Classification
    # movie_reviews = [
    #     "This movie was absolutely wonderful. The storyline was compelling and the characters were very realistic.",
    #     "I really loved the film! The plot had a few unexpected twists which kept me engaged till the end.",
    #     "The movie was alright. Not great, but not bad either. A decent one-time watch.",
    #     "I didn't enjoy the film that much. The plot was quite predictable and the characters lacked depth.",
    #     "This movie was not to my taste. It felt too slow and the storyline wasn't engaging enough.",
    #     "The film was okay. It was neither impressive nor disappointing. It was just fine.",
    #     "I was blown away by the movie! The cinematography was excellent and the performances were top-notch.",
    #     "I didn't like the movie at all. The story was uninteresting and the acting was mediocre at best.",
    #     "The movie was decent. It had its moments but was not consistently engaging."
    # ]
    # movie_review_labels = [
    #     "positive", 
    #     "positive", 
    #     "neutral", 
    #     "negative", 
    #     "negative", 
    #     "neutral", 
    #     "positive", 
    #     "negative", 
    #     "neutral"
    # ]
    # new_movie_reviews = [
    #     # A positive review
    #     "The movie was fantastic! I was captivated by the storyline from beginning to end.",
    #     # A negative review
    #     "I found the film to be quite boring. The plot moved too slowly and the acting was subpar.",
    #     # A neutral review
    #     "The movie was okay. Not the best I've seen, but certainly not the worst."
    # ]
    # clf = ZeroShotGPTClassifier(openai_model="llama3-8b-8192")
    # # Train the model 
    # clf.fit(X=movie_reviews, y=movie_review_labels)  
    # # Use the trained classifier to predict the sentiment of the new reviews
    # predicted_movie_review_labels = clf.predict(X=new_movie_reviews)  
    # for review, sentiment in zip(new_movie_reviews, predicted_movie_review_labels):
    #     print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n\n")



    ## Zero-Shot Text Classification
    # Load a demo dataset
    print("Zero-Shot Text Classification")
    X, y = get_classification_dataset() # labels: positive, negative, neutral
    print(f"X len is {len(X)}")
    print(f"y len is {len(y)}")
    clf = ZeroShotGPTClassifier(openai_model="llama3-8b-8192")
    clf.fit(X,y)
    clf.predict(X) # 實際上並沒有調整 GPT的權重，只是配置一些必要的設定
    print(clf.score(X,y)) # 使用 GPT內建知識進行推理


    ## Few-Shot Text Classification
    print("Few-Shot Text Classification")
    from skllm.models.gpt.classification.few_shot import (
    FewShotGPTClassifier
    )
    from skllm.datasets import (
    get_classification_dataset
    )


    # single label
    X, y = get_classification_dataset()
    clf = FewShotGPTClassifier(model="llama3-8b-8192")
    clf.fit(X,y)
    labels = clf.predict(X)
    print(clf.score(X,y))




    ## Text Translation
    print("Text Translation")
    translator = GPTTranslator()
    text_to_translate = ["Je suis étudiant en spécialisation en ingénierie de l'information"]
    # "I am happy that you are reading this post."
    translated_text = translator.fit_transform(text_to_translate)

    print(
        f"Text in French: \n{text_to_translate[0]}\n\nTranslated text in English: {translated_text[0]}"
    )










    ## Text Summarization
    print("Text Summarization")
    reviews = [
    "I dined at The Gourmet Kitchen last night and had a wonderful experience. The service was impeccable, the food was exquisite, and the ambiance was delightful. I had the seafood pasta, which was cooked to perfection. The wine list was also quite impressive. I would highly recommend this restaurant to anyone looking for a fine dining experience.",
    "I visited The Burger Spot for lunch today and was pleasantly surprised. Despite being a fast food joint, the quality of the food was excellent. I ordered the classic cheeseburger and it was juicy and flavorful. The fries were crispy and well-seasoned. The service was quick and the staff was friendly. It's a great place for a quick and satisfying meal.",
    "The Coffee Corner is my favorite spot to work and enjoy a good cup of coffee. The atmosphere is relaxed and the coffee is always top-notch. They also offer a variety of pastries and sandwiches. The staff is always welcoming and the service is fast. I enjoy their latte and the blueberry muffin is a must-try."
]
    gpt_summarizer = GPTSummarizer()
    summaries = gpt_summarizer.fit_transform(reviews)
    for review,summm in zip(reviews,summaries):
        print(f"Review: {review}\nSummary: {summm}\n\n")


    
    # ## tagging
    # print("Named Entity Recognition")
    # entities = {
    # "PERSON": "A name of an individual.",
    # "ORGANIZATION": "A name of a company.",
    # "DATE": "A specific time reference."
    # }

    # data = [
    # "Tim Cook announced new Apple products in San Francisco on June 3, 2022.",
    # "Elon Musk visited the Tesla factory in Austin on January 10, 2021.",
    # "Mark Zuckerberg introduced Facebook Metaverse in Silicon Valley on May 5, 2023."
    # ]

    # ner = GPTExplainableNER(entities=entities, display_predictions=True)
    # tagged = ner.fit_transform(data)
    # for d, t in zip(data, tagged):
    #     print(f"Data: {d}\nTagged: {t}\n\n")

