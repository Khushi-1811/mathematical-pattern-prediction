
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved models
@st.cache_resource
def load_models():
    next_value_model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/Project-1/Models/next_value_prediction_model-4.keras")
    type_classification_model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/Project-1/Models/sequence_classification_model.keras")
    coefficient_model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/Project-1/Models/coefficient_prediction_model-2.keras")
    return next_value_model, type_classification_model, coefficient_model

# Load preprocessing scalers
@st.cache_data
def load_preprocessed_data():
    with open("/content/drive/MyDrive/Colab Notebooks/Project-1/Scalers/preprocessed_data.pkl", "rb") as f:
      preprocessed_data = pickle.load(f)
    return preprocessed_data

# Load models and preprocessed data
next_value_model, type_classification_model, coefficient_model = load_models()
preprocessed_data = load_preprocessed_data()

# Extract scalers
seq_scaler = preprocessed_data["seq_scaler"]
next_scaler = preprocessed_data["next_scaler"]
coeff_scaler = preprocessed_data["coeff_scaler"]
label_encoder = preprocessed_data["label_encoder"]

# Streamlit UI
st.title("Mathematical Sequence Predictor")
st.write("Enter a sequence of numbers to predict the next value, sequence type, and equation coefficients.")

# User input
user_input = st.slider("Number of elements in sequence:", min_value=3, max_value=20, value=5)

# Collect user input dynamically
user_sequence = []
for i in range(user_input):
    num = st.number_input(f"Enter number {i+1}:", value=10 * (i + 1), step=1)
    user_sequence.append(num)

if st.button("Predict"):
    try:
        # Convert input to numpy array and reshape
        user_sequence = np.array(user_sequence).reshape(1, -1)

        # Preprocess the input
        max_seq_length = 20  # Should match training data
        user_padded = pad_sequences(user_sequence, maxlen=max_seq_length, padding="post")
        user_padded_scaled = seq_scaler.transform(user_padded)

        # Reshape for model
        user_padded_scaled = np.expand_dims(user_padded_scaled, axis=-1)

        # Predict next value
        next_value_scaled = next_value_model.predict(user_padded_scaled)
        try:
            next_value_transformed = next_scaler.inverse_transform(next_value_scaled)
        except ValueError as e:
            print(f"Error in inverse transformation: {e}")
            next_value_transformed = next_value_scaled

        next_value_transformed = np.clip(next_value_transformed, a_min=0, a_max=None)
        next_value_original = np.expm1(next_value_transformed)

        next_value_final = np.round(next_value_original)

        type_pred = type_classification_model.predict(user_padded_scaled)
        predicted_type = label_encoder.inverse_transform([np.argmax(type_pred)])[0]

        coeff_pred_scaled = coefficient_model.predict(user_padded_scaled)
        coeff_pred = np.round(coeff_scaler.inverse_transform(coeff_pred_scaled)[0])

        # Display Results
        st.subheader("Predictions:")
        st.write(f"**Predicted Next Value:** {next_value_final}")
        st.write(f"**Predicted Sequence Type:** {predicted_type}")
        st.write(f"**Predicted Coefficients:** {coeff_pred}")

    except Exception as e:
        st.error(f"Error processing input: {e}")
