from datasets import load_dataset
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

# lemmatizare
# initializare lemmatizer
lemmatizer = WordNetLemmatizer()

# functie pentru lemmatizare token-uri
def lemmatize_tokens(tokens):
    # conversie din pos in wordnet
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    # lematizare token-uri
    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    
    # returnare ca o lista
    return lemmas

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', None)

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# citirea datasetului

train_dataset = load_dataset("Dataset/split", split="train")
valid_dataset = load_dataset("Dataset/split", split="validation")
test_dataset  = load_dataset("Dataset/split", split="test")

df_train = pd.DataFrame(train_dataset)
df_valid = pd.DataFrame(valid_dataset)
df_test = pd.DataFrame(test_dataset)

# preprocesarea Datelor
# eliminarea duplicatelor

df_train = df_train.drop_duplicates(keep='first')
df_valid = df_valid.drop_duplicates(keep='first')
df_test = df_test.drop_duplicates(keep='first')

# df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
# transformarea din uppercase in lowercase

df_train = df_train.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_valid = df_valid.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_test = df_test.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)
# eliminare a caracterelor care nu sunt cuvinte si care nu sunt whitespace-uri

df_train = df_train.replace(to_replace=r'[^\w\s]', value='', regex=True)
df_valid = df_valid.replace(to_replace=r'[^\w\s]', value='', regex=True)
df_test = df_test.replace(to_replace=r'[^\w\s]', value='', regex=True)

# df = df.replace(to_replace=r'\d', value='', regex=True)
# eliminare cifre din text
# posibil sa nu mai fie nevoie daca am eliminat caracterele care nu sunt
# cuvinte si whitespace-uri

df_train = df_train.replace(to_replace=r'\d', value='', regex=True)
df_valid = df_valid.replace(to_replace=r'\d', value='', regex=True)
df_test = df_test.replace(to_replace=r'\d', value='', regex=True)

# tokenizare
# df['Message'] = df['Message'].apply(word_tokenize)

df_train['text'] = df_train['text'].apply(lambda x: tokenizer.tokenize(x) if isinstance(x, str) else x)
df_valid['text'] = df_valid['text'].apply(lambda x: tokenizer.tokenize(x) if isinstance(x, str) else x)
df_test['text'] = df_test['text'].apply(lambda x: tokenizer.tokenize(x) if isinstance(x, str) else x)

# stergere stopwords
# stop_words = set(stopwords.words('english'))
# df['Message'] = df['Message'].apply(lambda x: [word for word in x if word not in stop_words])

df_train['text'] = df_train['text'].apply(lambda x: [word for word in x if word not in stop_words])
df_valid['text'] = df_valid['text'].apply(lambda x: [word for word in x if word not in stop_words])
df_test['text'] = df_test['text'].apply(lambda x: [word for word in x if word not in stop_words])

# lemmatizare

df_train['lemmatized_text'] = df_train['text'].apply(lemmatize_tokens)
df_valid['lemmatized_text'] = df_valid['text'].apply(lemmatize_tokens)
df_test['lemmatized_text'] = df_test['text'].apply(lemmatize_tokens)

# TF-IDF vectorization
# concatenare token-uri in string

# print(df_train)
# print(df_valid)
# print(df_test)

df_train['lemmatized_str'] = df_train['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))
df_valid['lemmatized_str'] = df_valid['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))
df_test['lemmatized_str'] = df_test['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(df_train['lemmatized_str'])
Y_train = df_train['label']

X_valid_tfidf = tfidf.fit_transform(df_valid['lemmatized_str'])
Y_valid = df_valid['label']

X_test_tfidf = tfidf.fit_transform(df_test['lemmatized_str'])
Y_test = df_test['label']

print(X_train_tfidf)
print(Y_train)