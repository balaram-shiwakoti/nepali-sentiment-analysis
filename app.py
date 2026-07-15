# app.py
import streamlit as st
import pickle

# सेभ गरिएको मोडल लोड गर्ने
@st.cache_resource
def load_model():
    with open("nepali_sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# UI सेटअप
st.set_page_config(page_title="भावना-बुझ NLP", page_icon="🧠")
st.title("🧠 भावना-बुझ: नेपाली सेन्टिमेन्ट एनालाइजर")
st.write("तपाईंको नेपाली वाक्य यहाँ लेख्नुहोस् र यसले सकारात्मक, नकारात्मक वा तटस्थ के भाव व्यक्त गर्छ, पत्ता लगाउनुहोस्।")

# प्रयोगकर्ताको इनपुट
user_input = st.text_area("नेपालीमा केही लेख्नुहोस्:", placeholder="जस्तै: मलाई यो सेवा एकदमै मनपर्यो...")

if st.button("भावना जाँच्नुहोस्"):
    if user_input.strip() == "":
        st.warning("कृपया पहिले केही टेक्स्ट लेख्नुहोस्।")
    else:
        # Prediction गर्ने
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        
        # नतिजा देखाउने
        st.write("---")
        if prediction == 1:
            st.success("😊 **सकारात्मक प्रतिक्रिया (Positive Sentiment)**")
        elif prediction == -1:
            st.error("😡 **नकारात्मक प्रतिक्रिया (Negative Sentiment)**")
        else:
            st.info("😐 **तटस्थ प्रतिक्रिया (Neutral Sentiment)**")
            
        # थप रोमान्चक जानकारी (Confidence score)
        st.write("### थप विवरण:")
        st.write(f"• नकारात्मक हुने सम्भावना: {probabilities[0]*100:.2f}%")
        st.write(f"• तटस्थ हुने सम्भावना: {probabilities[1]*100:.2f}%")
        st.write(f"• सकारात्मक हुने सम्भावना: {probabilities[2]*100:.2f}%")
