# train_nlp.py
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# १. नेपाली Stopwords को सूची
NEPALI_STOPWORDS = [
    "र", "अनि", "पनि", "त", "तर", "म", "हामी", "तपाईं", "तिमी", "उनी", "यो", "त्यो", "यी", "ती",
    "को", "का", "की", "मा", "लाई", "ले", "बाट", "देखि", "द्वारा", "भने", "भनेर", "भए", "भयो", 
    "छ", "छन्", "हो", "हुन्", "थियो", "थी", "थिए", "गरे", "गर्ने", "जब", "तब", "जुन", "जस्तै"
]

# २. नमुना नेपाली डाटासेट
data = {
    "text": [
        "मलाई यो सामान एकदमै मनपर्यो, धेरै राम्रो छ!",
        "उत्पादनको गुणस्तर एकदमै खराब छ, पैसा खेर गयो।",
        "यो ठिकै छ, सामान्य प्रयोगको लागि उपयोगी हुन सक्छ।",
        "डेलिभरी एकदम छिटो भयो, म धेरै खुसी छु।",
        "कति नराम्रो सेवा! फोन पनि उठाउँदैनन्।",
        "कपडाको रङ राम्रो छ तर साइज अलि सानो भयो।",
        "मलाई यो कत्ति पनि मन परेन, फिर्ता करना चाहन्छु।",
        "धेरै राम्रो सेवा र गुणस्तरीय सामान!",
        "नयाँ अपडेट आएपछि एप चल्नै छाड्यो, बेकार छ।",
        "नेपालमा बनेको यो सामान निकै राम्रो लाग्यो।"
    ],
    "sentiment": [1, -1, 0, 1, -1, 0, -1, 1, -1, 1]
}

df = pd.DataFrame(data)

# ३. नेपाली युनिकोड र मात्रा सुरक्षित राख्ने Token Pattern र Vectorizer को सिर्जना
# token_pattern=r"(?u)\b\w+\b" ले साना मात्राहरूलाई बिगार्न दिँदैन
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r"(?u)\b\w+\b"
)

# ४. Stopwords लाई Vectorizer को नियम अनुसार पहिले नै सफा (Analyze) गर्ने
# यसले गर्दा 'UserWarning' पूर्ण रूपमा हराउँछ
analyze = vectorizer.build_analyzer()
cleaned_stopwords = list(set([token for word in NEPALI_STOPWORDS for token in analyze(word)]))

# ५. नयाँ Cleaned Stopwords सहित Pipeline बनाउने
model_pipeline = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 2), 
        token_pattern=r"(?u)\b\w+\b",
        stop_words=cleaned_stopwords
    ),
    MultinomialNB()
)

# ६. मोडल ट्रेनिङ
print("🧹 सफा गरिएको नेपाली Stopwords प्रयोग गरी मोडल ट्रेनिङ सुरु हुँदैछ...")
model_pipeline.fit(df["text"], df["sentiment"])
print("🎉 सफलतापूर्वक ट्रेनिङ सम्पन्न भयो (कुनै वार्निङ बिना)!")

# ७. मोडललाई सेभ गर्ने
with open("nepali_sentiment_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("💾 'nepali_sentiment_model.pkl' फाइल सफलतापूर्वक अपडेट गरियो।")
