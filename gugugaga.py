
import re
from collections import Counter
import langid
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import spacy
import string
from textblob import TextBlob
from transformers import MarianMTModel, MarianTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

def word_frequency(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return Counter(words)



#1
teksts = ("Mākoņainā dienā kaķis sēdēja uz palodzes. "
            "Kaķis domāja, kāpēc debesis ir pelēkas. "
            "Kaķis gribēja redzēt sauli, bet saule slēpās aiz mākoņiem.")
frekvences = word_frequency(teksts)
for vārds, skaits in frekvences.items():
    print(f"{vārds}: {skaits}")


    
#2

texts = [
    "Šodien ir saulaina diena.",
    "Today is a sunny day.",
    "Сегодня солнечный день."
]
for text in texts:
    language, confidence = langid.classify(text)
    print(f"Teksts: \"{text}\" - Valoda: {language}")



#3

def preprocess_text(text, lang='latvian'):
    words = word_tokenize(text.lower())
    
    words = [word for word in words if word not in string.punctuation]
    
    latvian_stopwords = {'ir', 'un', 'to', 'klāj', 'aiz', 'bet'}
    stop_words = set(latvian_stopwords)
    words = [word for word in words if word not in stop_words]
    
    return words

def calculate_similarity(text1, text2, lang='latvian'):
    words1 = preprocess_text(text1, lang)
    words2 = preprocess_text(text2, lang)
    
    set1 = set(words1)
    set2 = set(words2)
    
    common_words = set1.intersection(set2)
    
    total_unique_words = set1.union(set2)
    similarity_percentage = (len(common_words) / len(total_unique_words)) * 100
    
    return common_words, similarity_percentage

text1 = "Rudens lapas ir dzeltenas un oranžas. Lapas klāj zemi un padara to krāsainu."
text2 = "Krāsainas rudens lapas krīt zemē. Lapas ir oranžas un dzeltenas."

common, percentage = calculate_similarity(text1, text2)

print(f"Sakritīgie vārdi: {', '.join(common)}")
print(f"Sakritības līmenis: {percentage:.2f}%")



#4

def analyze_sentiment(sentence):
    translations = {
        "Šis produkts ir lielisks, esmu ļoti apmierināts!": "This product is great, I am very satisfied!",
        "Esmu vīlies, produkts neatbilst aprakstam.": "I am disappointed, the product does not match the description.",
        "Neitrāls produkts, nekas īpašs.": "Neutral product, nothing special."
    }
    english_sentence = translations[sentence]
    
    blob = TextBlob(english_sentence)
    polarity = blob.sentiment.polarity  

    print(polarity)
    
    if polarity > 0.5:
        sentiment = "pozitīvs"
    elif polarity < 0:
        sentiment = "negatīvs"
    else:
        sentiment = "neitrāls"
    
    return sentiment

sentences = [
    "Šis produkts ir lielisks, esmu ļoti apmierināts!",
    "Esmu vīlies, produkts neatbilst aprakstam.",
    "Neitrāls produkts, nekas īpašs."
]

for sentence in sentences:
    sentiment = analyze_sentiment(sentence)
    print(f"Teikums: \"{sentence}\" - Noskaņojums: {sentiment}")



#5

def clean_text(text):
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

raw_text = "@John: Šis ir lielisks produkts!!! Vai ne? 👏👏👏 http://example.com"
cleaned_text = clean_text(raw_text)
print("Cleaned Text:", cleaned_text)



#6

def summarize_text(article, max_length=50, min_length=25):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

article = """
Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir piekļuve Baltijas jūrai. Tā ir viena no Eiropas Savienības dalībvalstīm.
"""

summary = summarize_text(article)
print("Summary:", summary)


#7


model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

words = ["māja", "dzīvoklis", "jūra"]

embeddings = model.encode(words, convert_to_tensor=True)

similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

print("Vārdu līdzības:")
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        sim = similarity_matrix[i][j].item()
        print(f"Līdzība starp '{words[i]}' un '{words[j]}': {sim:.4f}")


#8
nlp = spacy.load("xx_ent_wiki_sm") 

text = "Valsts prezidents Egils Levits piedalījās pasākumā, ko organizēja Latvijas Universitāte."

doc = nlp(text)

person_entities = []
org_entities = []

for ent in doc.ents:
    if ent.label_ == "PER":
        if ent.text.endswith("Universitāte"):
            org_entities.append(ent.text)
        else:
            person_entities.append(ent.text)
    elif ent.label_ == "ORG":
        org_entities.append(ent.text)

print("Personvārdi:")
for person in person_entities:
    print(f"- {person}")

print("\nOrganizācijas:")
for org in org_entities:
    print(f"- {org}")

#9
def generate_story(starting_phrase, max_length=50, num_return_sequences=1):
    generator = pipeline("text-generation", model="gpt2")
    
    generated_texts = generator(
        starting_phrase, 
        max_length=max_length, 
        num_return_sequences=num_return_sequences,
        do_sample=True, 
        top_k=50,      
        top_p=0.95      
    )
    
    return [text['generated_text'] for text in generated_texts]

starting_phrase = "Reiz kādā tālā zemē..."

generated_stories = generate_story(starting_phrase, max_length=50)

for i, story in enumerate(generated_stories):
    print(f"Stāsts {i + 1}: {story}")



#10

def translate(texts, src_lang="lv", tgt_lang="en"):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    translated = model.generate(**tokenizer(texts, return_tensors="pt", padding=True))
    
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return translated_texts

latv_texts = [
    "Labdien! Kā jums klājas?",
    "Es šodien lasīju interesantu grāmatu."
]

anglu_texts = translate(latv_texts)

for lv, en in zip(latv_texts, anglu_texts):
    print(f"Latviešu: {lv}")
    print(f"Angļu: {en}\n")
