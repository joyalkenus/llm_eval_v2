# User Preference Predictor

This Python script evaluates user content preferences using OpenAI's GPT-4 model. It processes a dataset of user interactions with posts and predicts user preferences for new content based on training data patterns.

## Features

- Loads and processes a dataset of 15,000 samples
- Splits data into training (400 samples) and validation (400 samples) sets
- Uses OpenAI's GPT-4 to learn user preferences from training data
- Predicts preferences for validation data in batches
- Calculates accuracy and F1 score metrics

## Prerequisites

- Python 3.x
- OpenAI API key
- Required Python packages:
  - pandas
  - scikit-learn
  - openai

## Installation

1. Clone the repository
2. Install required packages: 