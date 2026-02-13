import streamlit as st
import pickle
import pandas as pd
import os

# Set page configuration
st.set_page_config(page_title="World Cup Predictor", layout="centered")

st.title("⚽ World Cup Match Predictor")
st.write("Select two teams to predict the winner based on historical data.")

# Load model and encoder
@st.cache_resource
def load_artifacts():
    # Ensure these filenames match what you saved in your training script
    model_path = r'C:\Users\dell\Desktop\Anurag_sleep\sleep_night\sleep_discussion\model_serving\world_cup_model.pkl'
    encoder_path = r'C:\Users\dell\Desktop\Anurag_sleep\sleep_night\sleep_discussion\model_serving\team_encoder.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

if model is None or encoder is None:
    st.error("Error: Model or Encoder file not found. Please run 'train_world_cup.py' first to generate the .pkl files.")
else:
    # Get list of teams from the encoder
    teams = sorted(encoder.classes_)
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Select Home Team", teams)
        
    with col2:
        away_team = st.selectbox("Select Away Team", teams)
        
    if st.button("Predict Result"):
        if home_team == away_team:
            st.warning("Please select two different teams.")
        else:
            # Encode the selected teams
            # We wrap in list [] because transform expects an iterable
            home_encoded = encoder.transform([home_team])[0]
            away_encoded = encoder.transform([away_team])[0]
            
            # Prepare input for the model
            # Feature names must match training: ['HomeTeam_Enc', 'AwayTeam_Enc']
            input_data = pd.DataFrame([[home_encoded, away_encoded]], columns=['HomeTeam_Enc', 'AwayTeam_Enc'])
            
            # Predict
            prediction = model.predict(input_data)[0]
            
            # Map prediction to result (0: Draw, 1: Home Win, 2: Away Win)
            if prediction == 1:
                st.success(f"🏆 Prediction: {home_team} Wins!")
            elif prediction == 2:
                st.success(f"🏆 Prediction: {away_team} Wins!")
            else:
                st.info("⚖️ Prediction: Draw")
            
            st.write("---")