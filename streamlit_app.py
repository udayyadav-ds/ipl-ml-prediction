import streamlit as st
import requests
import pandas as pd

# Page configuration
st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="wide")

# App title and description
st.title("🏏 IPL Match Winner Predictor")
st.markdown("""
Predict the outcome of an IPL match based on historical data, team matchups, and venue context.
Specifically built for the **Punjab Kings vs Gujarat Titans** match today!
""")

# API endpoint (adjust based on your setup)
API_URL = "http://localhost:8000"

# Fetch metadata from API
@st.cache_data
def get_metadata( ):
    try:
        response = requests.get(f"{API_URL}/metadata")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error connecting to the backend API: {e}")
        return None

metadata = get_metadata()

if metadata:
    teams = metadata['teams']
    venues = metadata['venues']
    toss_decisions = metadata['toss_decisions']
    
    # User input sidebar
    st.sidebar.header("Match Details")
    
    # Pre-select Punjab Kings and Gujarat Titans for convenience
    default_team1 = "Punjab Kings" if "Punjab Kings" in teams else teams[0]
    default_team2 = "Gujarat Titans" if "Gujarat Titans" in teams else teams[1]
    
    team1 = st.sidebar.selectbox("Select Team 1", teams, index=teams.index(default_team1))
    team2 = st.sidebar.selectbox("Select Team 2", teams, index=teams.index(default_team2))
    
    # Ensure team1 and team2 are different
    if team1 == team2:
        st.sidebar.warning("Please select different teams.")
    
    venue = st.sidebar.selectbox("Select Venue", venues)
    
    toss_winner = st.sidebar.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.sidebar.selectbox("Toss Decision", toss_decisions)
    
    # Main prediction area
    if st.sidebar.button("Predict Winner"):
        # API call
        input_data = {
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision
        }
        
        try:
            with st.spinner("Analyzing match data..."):
                response = requests.post(f"{API_URL}/predict", json=input_data)
                
            if response.status_code == 200:
                result = response.json()
                winner = result['predicted_winner']
                probability = result['win_probability']
                all_probs = result['all_probabilities']
                
                # Display result
                st.success(f"### Predicted Winner: **{winner}**")
                st.write(f"Confidence Level: **{probability:.2%}**")
                
                # Visualization of probabilities
                st.write("#### Win Probabilities:")
                prob_df = pd.DataFrame({
                    'Team': list(all_probs.keys()),
                    'Probability': list(all_probs.values())
                })
                # Filter only for the two playing teams for clarity
                prob_df = prob_df[prob_df['Team'].isin([team1, team2])]
                st.bar_chart(prob_df.set_index('Team'))
                
                # Match Context
                st.info(f"Today's match between **{team1}** and **{team2}** at **{venue}** is predicted to favor **{winner}** based on our ML model.")
                
            else:
                st.error(f"Error from API: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Could not reach prediction service: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, FastAPI, and Scikit-Learn")
else:
    st.warning("Backend API is not reachable. Please ensure the FastAPI server is running.")
    st.info("Try running: `uvicorn main:app --reload` in your terminal.")
