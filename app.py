import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

sw = set(stopwords.words('english'))

# load model
model = pickle.load(open("model.pkl", "rb"))
tf = pickle.load(open("vectorizer.pkl", "rb"))

# clean text
def clean(t):
    t = t.lower()
    t = re.sub(r"http\\S+", "", t)
    t = re.sub(r"[^a-zA-Z]", " ", t)
    t = t.split()
    t = [i for i in t if i not in sw]
    return " ".join(t)

# prediction
def pred(t):
    t = clean(t)
    t = tf.transform([t])
    return model.predict(t)[0]

# ui
st.title("Twitter Sentiment Analysis")

txt = st.text_area("Enter Tweet")

if st.button("Predict"):

    ans = pred(txt)

    if ans == "Positive":
        st.success("😊 Positive Tweet")

    elif ans == "Negative":
        st.error("😠 Negative Tweet")

    else:
        st.warning("😐 Neutral Tweet")