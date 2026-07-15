# train_nlp.py
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# १. नमुना नेपाली डाटासेट (Sample Nepali Dataset)
# वास्तविक प्रोजेक्टमा हामी Kaggle को ठूलो dataset प्रयोग गर्न सक्छौँ।
data = {
    "text": [
        "मलाई यो सामान एकदमै मनपर्यो, धेरै राम्रो छ!",
        "उत्पादनको गुणस्तर एकदमै खराब छ, पैसा खेर गयो।",
        "यो ठिकै छ, सामान्य प्रयोगको लागि उपयोगी हुन सक्छ।",
        "डेलिभरी एकदम छिटो भयो, म धेरै खुसी छु।",
        "कति नराम्रो सेवा! फोन पनि उठाउँदैनन्।",
        "कपडाको रङ राम्रो छ तर साइज अलि सानो भयो।",
        "मलाई यो कत्ति पनि मन परेन, फिर्ता गर्न चाहन्छु।",
        "धेरै राम्रो सेवा र गुणस्तरीय सामान!",
        "नयाँ अपडेट आएपछि एप चल्नै छाड्यो, बेकार छ।",
        "नेपालमा बनेको यो सामान निकै राम्रो लाग्यो।"
    ],
    # १: सकारात्मक (Positive), -१: नकारात्मक (Negative), ०: तटस्थ (Neutral)
    "sentiment": [1, -1, 0, 1, -1, 0, -1, 1, -1, 1]
}

df = pd.DataFrame(data)

# २. TF-IDF र Naive Bayes को Pipeline बनाउने
# यसले नेपाली शब्दहरूलाई गणितीय नम्बरमा बदल्छ र वर्गीकरण गर्छ।
model_pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)), # एक वा दुई शब्दको समूहलाई चिन्ने
    MultinomialNB()
)

# ३. मोडल ट्रेनिङ
print("मोडल ट्रेनिङ सुरु हुँदैछ...")
model_pipeline.fit(df["text"], df["sentiment"])
print("ट्रेनिङ सम्पन्न भयो!")

# ४. मोडललाई पिकल (Pickle) फाइलमा सेभ गर्ने
with open("nepali_sentiment_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("मोडल 'nepali_sentiment_model.pkl' नाममा सुरक्षित गरियो।")
