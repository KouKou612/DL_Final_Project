import os
import numpy as np
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from model import FakeNewsModel
from train import ModelTrainer

def main():
    # File paths - update these according to your file locations
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    train_path = os.path.join(BASE_DIR,"..", "data/train.csv")
    test_path = os.path.join(BASE_DIR,"..",  "data/test.csv")
    eval_path = os.path.join(BASE_DIR,"..",  "data/evaluation.csv")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    data_loader = DataLoader()
    if not data_loader.load_data(train_path, test_path, eval_path):
        return
    
    train_df, test_df, eval_df = data_loader.get_data()
    y_train, y_test, y_eval = data_loader.get_labels()
    
    print("\n--- Fake news count ---")
    print(f"Training set: {np.sum(y_train)} / {len(y_train)}")
    print(f"Test set: {np.sum(y_test)} / {len(y_test)}")
    print(f"Evaluation set: {np.sum(y_eval)} / {len(y_eval)}")

    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    train_texts_processed = preprocessor.preprocess_texts(train_df['combined_text'].values)
    test_texts_processed = preprocessor.preprocess_texts(test_df['combined_text'].values)
    eval_texts_processed = preprocessor.preprocess_texts(eval_df['combined_text'].values)
    
    # Create sequences
    X_train, X_test, X_eval = preprocessor.create_sequences(
        train_texts_processed, 
        test_texts_processed, 
        eval_texts_processed,
        max_features=10000,
        max_len=500
    )
    
    # Step 3: Build model
    print("\nStep 3: Building model...")
    vocab_size = preprocessor.get_vocab_size()
    model_builder = FakeNewsModel(vocab_size=vocab_size, max_length=500)
    model = model_builder.build_cnn_model()
    model_builder.get_model_summary()
    
    # Step 4: Train model
    print("\nStep 4: Training model...")
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_eval, y_eval, epochs=10, batch_size=32)
    
    # Step 5: Evaluate model
    print("\nStep 5: Evaluating model...")
    y_pred, y_pred_proba = trainer.evaluate(X_test, y_test)
    
    # Step 6: Visualize results
    print("\nStep 6: Visualizing results...")
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(y_test, y_pred)
    
    # Step 7: Save model
    print("\nStep 7: Saving model...")
    model_path = os.path.join(MODEL_DIR, "fake_news_cnn_model.h5")
    model_builder.save_model(model_path)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()