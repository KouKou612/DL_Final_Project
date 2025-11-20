import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from data_loader import *

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab')
except:
    print("NLTK downloads completed or error occurred")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = None
        self.max_sequence_length = 0
        
    def clean_text(self, text):
        if isinstance(text, float) or text is None:
            return ""
         
        # Convert to lowercase
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        if not text or text == "":
            return ""
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def preprocess_texts(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        processed_texts = [self.tokenize_and_lemmatize(text) for text in cleaned_texts]
        return processed_texts
    
    def create_sequences(self, train_texts, test_texts, eval_texts, max_features=10000, max_len=500):  
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
        
        # Fit on training texts
        self.tokenizer.fit_on_texts(train_texts)
        
        # Convert texts to sequences
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)
        eval_sequences = self.tokenizer.texts_to_sequences(eval_texts)
        
        # Pad sequences
        X_train = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
        X_test = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
        X_eval = pad_sequences(eval_sequences, maxlen=max_len, padding='post', truncating='post')
        
        self.max_sequence_length = max_len
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Training sequences shape: {X_train.shape}")
        print(f"Test sequences shape: {X_test.shape}")
        print(f"Evaluation sequences shape: {X_eval.shape}")
        
        return X_train, X_test, X_eval
    
    def get_vocab_size(self):
        if self.tokenizer:
            return len(self.tokenizer.word_index)
        return 0
    
    