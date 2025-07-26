# Speech-Emotion-Recognition

This project implements a Speech Emotion Recognition (SER) system using deep learning to classify human emotions from audio signals. It leverages MFCC (Mel Frequency Cepstral Coefficients) feature extraction and a Convolutional Neural Network (CNN) to identify emotions such as happy, sad, angry, and neutral from recorded speech.

## Features

- Extracts MFCC features from WAV audio files
- Preprocesses and normalizes input audio data
- Trains a CNN model on labeled emotion audio datasets
- Supports evaluation with accuracy, loss plots, and confusion matrix
- Includes visualizations for waveform and spectrogram analysis
- Predicts the emotion from unseen speech input

## Use Cases

- Human-computer interaction improvement
- Emotion-aware virtual assistants
- Call center sentiment analysis
- AI-powered therapy or well-being tracking tools

## Project Structure

- `Speech Emotion Recognition - Sound Classification.ipynb`: Jupyter notebook with full code and workflow
- Audio dataset: Standard labeled speech dataset (e.g., RAVDESS, TESS)
- Feature extraction using LibROSA
- CNN-based model for classification

## Technologies Used

- Python
- NumPy, Pandas
- LibROSA
- Matplotlib, Seaborn
- Keras / TensorFlow (for CNN model)

## Performance

The CNN model is trained on preprocessed MFCC features and achieves high accuracy in classifying emotions across multiple categories, depending on dataset size and augmentation.

---

