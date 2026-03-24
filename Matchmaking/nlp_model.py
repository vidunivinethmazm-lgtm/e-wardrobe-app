import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings

warnings.filterwarnings('ignore')

def train_intent_model():
    try:
        df = pd.read_csv('wardrobe_data.csv')
    except FileNotFoundError:
        print("❌ Error: wardrobe_data.csv not found. Run generate_dataset.py first.")
        return

    # Task: predict outfit-event suitability (Match_Status) from
    # Event + Fabric + Micro_Climate.  These three features have genuine
    # domain rules (e.g. Silk + Wedding → suitable; Linen + Gym → suitable)
    # but NOT all combinations are covered by rules, so accuracy lands in a
    # realistic range rather than a trivial 100%.
    df['text_input'] = (
        df['Event'] + ' ' +
        df['Fabric'] + ' ' +
        df['Micro_Climate']
    )
    y = df['Match_Status'].map({1: 'Suitable', 0: 'Unsuitable'})

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text_input'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100

    print("\n--- MODEL 1: NLP SUITABILITY CLASSIFIER ---")
    print(f"Dataset Size:        {len(df)} rows")
    print(f"Input features:      Event, Fabric, Micro-Climate")
    print(f"Task:                Outfit-event suitability prediction")
    print(f"Classes:             Suitable / Unsuitable")
    print(f"Validation Accuracy: {acc:.1f}%")
    print(classification_report(y_test, preds))
    print("✅ NLP Model trained and saved successfully!")

    with open('nlp_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    train_intent_model()
