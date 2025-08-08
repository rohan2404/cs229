import numpy as np
import pandas as pd
from typing import List
import json
import html
import re

default_dictionary_path = "dictionary.json"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep = '\t', header=None, names = ['label', 'text'], encoding='utf-8')
    df['label'] = df['label'].map({"spam": 1, "ham": 0})
    def clean_text(s: str) -> str:
        s = html.unescape(s)                     # unescape HTML entities - idk gpt said to do this
        s = s.lower()                            # lowercase
        s = re.sub(r"[''`]", "", s)              # remove apostrophes without adding spaces
        s = re.sub(r"[^a-z\s]", " ", s)          # remove everything except letters and spaces (drops digits)
        s = re.sub(r"\s+", " ", s).strip()       # collapse whitespace
        return s
    df["text"]=df["text"].astype(str).apply(clean_text)
    return df

def get_words(message: str) -> List[str]:
    return [w for w in message.split() if w]

def create_dictionary(messages: pd.Series) -> dict:
    #splits strings into lists of words which is then exploded into a series of all words used
    words = messages.apply(get_words).explode() 
    #stores frequencies of words so we can then drop duplicates without losing info
    word_frequencies = words.value_counts().to_dict()
    words = words.drop_duplicates()
    words = words.astype(str)
    #creates a function which puts an NaN for frequencies <= 5 (so we can then drop the NaNs)
    def remove_lowfreqword(word: str):
        if word_frequencies[word] <=5: return np.nan
        else: return word
    words = words.apply(remove_lowfreqword)
    words = words.dropna().reset_index(drop=True)
    #map word -> index
    dictionary = {v:k for k,v in words.to_dict().items()}
    json.dump(dictionary, open(default_dictionary_path, 'w'))
    return dictionary

def load_dictionary(path: str) -> dict:
    return json.load(open(path))

def text_to_vector(text: str, dictionary: dict) -> np.ndarray:
    #maps frequency of word in a single message to its correct word index. This will be our feature vector
    words = get_words(text)
    words_series = pd.Series(words)
    words_freqs = words_series.value_counts()
    words_series = words_series.drop_duplicates()
    d = len(dictionary)
    vector = np.zeros(d)
    for x in words_series:
        if x in dictionary: #we'll ignore words not in our dictionary 
            vector[dictionary[x]] = words_freqs[x]
    return vector

def construct_matrix_data(messages: pd.Series, dictionary: dict) -> np.ndarray:
    d = len(dictionary)
    m = len(messages)
    matrix = np.empty((m,d))
    i=0
    for x in messages:
        matrix[i,:] = text_to_vector(x, dictionary)
        i+=1
    return matrix

def fit_naive_bayes(training_data: np.ndarray, training_labels: np.ndarray):
    #training data has shape (m,d)
    d = training_data.shape[1]
    number_of_words_y0: np.ndarray = training_data[training_labels == 0].sum(axis=0)
    number_of_words_y1: np.ndarray = training_data[training_labels == 1].sum(axis=0)
    #laplace smooth the fit - could probs make a hyperparameter out of this but it was a good fit so it don't matter
    phi_j_y0 = (number_of_words_y0 + 1)/(np.sum(number_of_words_y0) + d)
    phi_j_y1 = (number_of_words_y1 + 1)/(np.sum(number_of_words_y1) + d)
    #labels must take on value 1 for spam, 0 for non-spam
    phi_y1 = np.sum(training_labels)/len(training_labels) 
    return phi_j_y0, phi_j_y1, phi_y1

def make_prediction_nb(input_feature:np.ndarray, phi_j_y0: np.ndarray, phi_j_y1: np.ndarray, phi_y1: float) -> np.ndarray:
    log_p1 = np.log(phi_y1) + np.dot(input_feature, np.log(phi_j_y1))
    log_p0 = np.log(1 - phi_y1) + np.dot(input_feature, np.log(phi_j_y0))
    return np.where(log_p1>log_p0, 1, 0) #returns array of 1 when the class prob for spam is greater than non-spam.

def top_5_words_nb(phi_j_y0: np.ndarray, phi_j_y1: np.ndarray, dictionary: dict):
    z = np.log(phi_j_y1) - np.log(phi_j_y0)
    idx_top5 = np.argpartition(z, -5)[-5:] #only sorts the top 5 numbers 
    return [k for k,v in dictionary.items() if v in idx_top5]
    