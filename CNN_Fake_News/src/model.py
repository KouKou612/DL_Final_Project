import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

class FakeNewsModel:
    def __init__(self, vocab_size, embedding_dim=100, max_length=500):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
    
    def build_cnn_model(self):
        self.model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size + 1,  # +1 for padding
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                mask_zero=True
            ),
            
            # First Conv1D layer
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def get_model_summary(self):
        if self.model:
            return self.model.summary()
        return "Model not built yet."
    
    def save_model(self, filepath):
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save.")