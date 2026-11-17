# Deep Learning NLP Text Classifier — SMS Spam Detection

A TensorFlow/Keras neural network that classifies SMS messages as spam or ham.

## Techniques
- Keras TextVectorization for tokenization and integer vectorization
- Embedding layer and multi-layer Dense neural network
- Batch Normalization and Dropout
- Early Stopping
- Training/validation loss and accuracy curves
- Test-set evaluation, confusion matrix, and ROC-AUC
- Sample predictions with confidence values
- Saved Keras model

## Run in Google Colab
1. Upload and open `Deep_Learning_NLP_SMS_Spam_Classifier.ipynb`.
2. Select Runtime → Run all.
3. Wait for all cells to finish.
4. Download generated files from the Colab Files panel.

The first run needs internet access to download the SMS dataset.

## Dependencies
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## Generated files
- `sms_spam_text_classifier.keras`
- `training_curves.png`
- `test_evaluation_plots.png`
- `sample_inference_results.csv`
- `training_history.csv`

Run the notebook before submission so output files are generated from an actual training run.
