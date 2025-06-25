from datasets import load_dataset
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import exp
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def softmax(z):
    # z matrice (N, K), unde N = nr. de exemple, K = nr. de clase
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict_softmax(X, W):
    # X matrice (N, D) exemple
    # W matrice (D, K) ponderi clase 
    z = np.dot(X, W)
    return softmax(z)

def cross_entropy(Y_pred, Y_true):
    # Y_pred - probabilitatile prezise
    # Y_true - etichetele one-hot encoding
    N = Y_pred.shape[0]
    # interval [0, 1]
    epsilon = 1e-15
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    return -np.sum(Y_true * np.log(Y_pred)) / N

# acuratetea modelului
def accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=1)
    return np.mean(predictions == Y_true)

# functie care converteste etichetele in format one-hot
def one_hot_encode(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

# functie pentru antrenare si evaluare
def train_and_eval_logistic(X_train, T_train, X_test, T_test, X_valid, T_valid,
                            num_classes=6, lr = .01, epochs_no=1000, reg_lambda=.1):
    
    (N, D) = X_train.shape

    # initializare ponderi
    w = np.random.randn(D, num_classes) * 0.01

    T_train_onehot = one_hot_encode(T_train, num_classes)
    T_valid_onehot = one_hot_encode(T_valid, num_classes)
    T_test_onehot = one_hot_encode(T_test, num_classes)

    train_acc, test_acc, valid_acc = [], [], []
    train_loss, test_loss, valid_loss = [], [], []

    for epochs in range(epochs_no):
        
        # forward
        Y_train = predict_softmax(X_train, w)
        Y_valid = predict_softmax(X_valid, w)
        Y_test = predict_softmax(X_test, w)

        # calcul loss (cross-entropy)
        train_loss_val = cross_entropy(Y_train, T_train_onehot) + (reg_lambda / 2) * np.sum(w * w)
        valid_loss_val = cross_entropy(Y_valid, T_valid_onehot) + (reg_lambda / 2) * np.sum(w * w)
        test_loss_val = cross_entropy(Y_test, T_test_onehot) + (reg_lambda / 2) * np.sum(w * w)

        train_loss.append(train_loss_val)
        valid_loss.append(valid_loss_val)
        test_loss.append(test_loss_val)

        # calcul acuratete
        train_acc_val = accuracy(Y_train, T_train)
        valid_acc_val = accuracy(Y_valid, T_valid)
        test_acc_val = accuracy(Y_test, T_test)

        train_acc.append(train_acc_val)
        valid_acc.append(valid_acc_val)
        test_acc.append(test_acc_val)

        # backward - calcul gradient - softmax + cross-entropy
        grad = (1 / N) * X_train.T @ (Y_train - T_train_onehot) + reg_lambda * w

        # actualizare ponderi
        w = w - lr * grad

        if epochs % 10 == 0 or epochs == epochs_no - 1:
            print(f"Epoca {epochs:3d}: Train Loss={train_loss_val:.4f}, Train Acc={train_acc_val:.4f}, "
                  f"Val Loss={valid_loss_val:.4f}, Val Acc={valid_acc_val:.4f}")
    
    return w, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc

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

df_train = df_train.map(lambda x: x.lower() if isinstance(x, str) else x)
df_valid = df_valid.map(lambda x: x.lower() if isinstance(x, str) else x)
df_test = df_test.map(lambda x: x.lower() if isinstance(x, str) else x)

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

df_train['lemmatized_str'] = df_train['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))
df_valid['lemmatized_str'] = df_valid['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))
df_test['lemmatized_str'] = df_test['lemmatized_text'].apply(lambda tokens: ' '.join(tokens))

tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=3,             
    max_features=10000
)

X_train_tfidf = tfidf.fit_transform(df_train['lemmatized_str'])
Y_train = df_train['label']

X_valid_tfidf = tfidf.transform(df_valid['lemmatized_str'])
Y_valid = df_valid['label']

X_test_tfidf = tfidf.transform(df_test['lemmatized_str'])
Y_test = df_test['label']

# adaugare coloana de bias (coloana de 1)

X_train_bias = hstack([X_train_tfidf, csr_matrix(np.ones((X_train_tfidf.shape[0], 1)))])
X_valid_bias = hstack([X_valid_tfidf, csr_matrix(np.ones((X_valid_tfidf.shape[0], 1)))])
X_test_bias = hstack([X_test_tfidf, csr_matrix(np.ones((X_test_tfidf.shape[0], 1)))])

# conversie la array pentru dot product

X_train_bias = X_train_bias.toarray()
X_valid_bias = X_valid_bias.toarray()
X_test_bias = X_test_bias.toarray()

print("Antrenare model")

w, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = train_and_eval_logistic(
        X_train_bias, Y_train, X_test_bias, Y_test, X_valid_bias, Y_valid,
        num_classes=6, lr=0.01, epochs_no=200, reg_lambda=0.1
    )

Y_train_final = predict_softmax(X_train_bias, w)
Y_val_final = predict_softmax(X_valid_bias, w)
Y_test_final = predict_softmax(X_test_bias, w)
        
train_pred = np.argmax(Y_train_final, axis=1)
val_pred = np.argmax(Y_val_final, axis=1)
test_pred = np.argmax(Y_test_final, axis=1)
        
print(f"Acuratete finala pe antrenament: {train_acc[-1]:.4f}")
print(f"Acuratete finala pe validare: {val_acc[-1]:.4f}")
print(f"Acuratete finala pe test: {test_acc[-1]:.4f}")
print()

print("Antrenare model folosind biblioteca")

model = LogisticRegression(
    multi_class='multinomial', 
    solver='lbfgs', 
    max_iter=1000,
    C=1/0.1,
    verbose=1
)

model.fit(X_train_bias, Y_train)

train_pred = model.predict(X_train_bias)
valid_pred = model.predict(X_valid_bias)
test_pred = model.predict(X_test_bias)

train_acc = accuracy_score(Y_train, train_pred)
valid_acc = accuracy_score(Y_valid, valid_pred)
test_acc = accuracy_score(Y_test, test_pred)

print(f"Acuratete finala pe antrenament: {train_acc:.4f}")
print(f"Acuratete finala pe validare: {valid_acc:.4f}")
print(f"Acuratete finala pe test: {test_acc:.4f}")