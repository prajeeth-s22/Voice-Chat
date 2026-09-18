"""
train_model.py
==============
Training script for the Voice-Enabled Chatbot intent classification model.

Architecture:
  Input (TF-IDF) → Dense(128, ReLU) → Dropout(0.35) → Dense(64, ReLU) → Dropout(0.25) → Output(Softmax)

Usage:
  python train_model.py

Outputs:
  - chatbot_model.keras     : Trained Keras model
  - vectorizer.pkl          : Fitted TF-IDF vectorizer
  - label_encoder.pkl       : Fitted label encoder
  - training_history.png    : Training accuracy/loss plot
"""

import json
import pickle
import numpy as np
import os

# Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_intents(filepath='intents.json'):
    """Load intents dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['intents']


def prepare_data(intents):
    """Extract patterns and labels from intents data."""
    patterns = []
    labels = []

    for intent in intents:
        tag = intent['tag']
        for pattern in intent['patterns']:
            patterns.append(pattern.lower().strip())
            labels.append(tag)

    print(f"Total training patterns: {len(patterns)}")
    print(f"Total intent classes:    {len(set(labels))}")
    print(f"Intent classes:          {sorted(set(labels))}")
    print()

    return patterns, labels


def create_tfidf_features(patterns):
    """Create TF-IDF feature matrix from text patterns."""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True
    )
    X = vectorizer.fit_transform(patterns).toarray()
    print(f"TF-IDF feature matrix shape: {X.shape}")
    return X, vectorizer


def encode_labels(labels):
    """Encode string labels to one-hot vectors."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    y_onehot = keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    print(f"Label classes: {list(label_encoder.classes_)}")
    print(f"One-hot shape: {y_onehot.shape}")
    return y_onehot, label_encoder, num_classes


def build_model(input_dim, num_classes):
    """Build the Keras neural network for intent classification."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     name='dense_1'),
        layers.Dropout(0.35, name='dropout_1'),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     name='dense_2'),
        layers.Dropout(0.25, name='dropout_2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history, filepath='training_history.png'):
    """Generate and save training accuracy/loss plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#4fc3f7', linewidth=2)
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff6b6b', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss', color='#4fc3f7', linewidth=2)
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Val Loss', color='#ff6b6b', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nTraining plot saved to: {filepath}")

    except ImportError:
        print("\nmatplotlib not installed — skipping training plot generation.")
        print("Install with: pip install matplotlib")


def main():
    print("=" * 60)
    print("  Voice-Enabled Chatbot — Model Training")
    print("=" * 60)
    print()

    # 1. Load intents
    print("[1/7] Loading intents dataset...")
    intents = load_intents()
    print(f"      Loaded {len(intents)} intent categories.\n")

    # 2. Prepare data
    print("[2/7] Preparing training data...")
    patterns, labels = prepare_data(intents)

    # 3. TF-IDF features
    print("[3/7] Creating TF-IDF features...")
    X, vectorizer = create_tfidf_features(patterns)
    print()

    # 4. Encode labels
    print("[4/7] Encoding labels...")
    y, label_encoder, num_classes = encode_labels(labels)
    print()

    # 5. Train/validation split
    print("[5/7] Splitting data (80% train / 20% validation)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"      Training samples:   {X_train.shape[0]}")
    print(f"      Validation samples: {X_val.shape[0]}")
    print()

    # 6. Build and train model
    print("[6/7] Building and training neural network...")
    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()
    print()

    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )

    # Final metrics
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print()
    print("-" * 40)
    print(f"  Final Training Accuracy:   {train_acc:.4f}")
    print(f"  Final Training Loss:       {train_loss:.4f}")
    print(f"  Final Validation Accuracy: {val_acc:.4f}")
    print(f"  Final Validation Loss:     {val_loss:.4f}")
    print("-" * 40)
    print()

    # 7. Save model and artifacts
    print("[7/7] Saving model and artifacts...")

    model.save('chatbot_model.keras')
    print("      Saved: chatbot_model.keras")

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("      Saved: vectorizer.pkl")

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("      Saved: label_encoder.pkl")

    # Generate training plot
    plot_training_history(history)

    print()
    print("=" * 60)
    print("  Training complete! Model is ready for deployment.")
    print("=" * 60)


if __name__ == '__main__':
    main()
