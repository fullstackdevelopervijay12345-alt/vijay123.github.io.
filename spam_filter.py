import os
import re
import string
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_NAME = "spam.csv"

print("====================================")
print("EMAIL SPAM FILTER")
print("====================================")


if not os.path.exists(FILE_NAME):
    print("ERROR: spam.csv not found!")
    exit()


try:
    df = pd.read_csv(
        FILE_NAME,
        encoding="utf-8"
    )

except UnicodeDecodeError:
    df = pd.read_csv(
        FILE_NAME,
        encoding="latin-1"
    )


print("\nDataset loaded successfully!")

print("Columns:")
print(df.columns.tolist())

print("\nNumber of records:", len(df))


# ============================================================
# 2. CHECK COLUMNS
# ============================================================

if "label" not in df.columns or "message" not in df.columns:

    print("\nERROR: CSV must contain:")
    print("label,message")
    exit()


df = df[["label", "message"]]


# ============================================================
# 3. REMOVE EMPTY VALUES
# ============================================================

df = df.dropna()

df["label"] = df["label"].astype(str)

df["message"] = df["message"].astype(str)


# ============================================================
# 4. CONVERT LABELS
# ============================================================

df["label"] = (
    df["label"]
    .str.lower()
    .str.strip()
)


df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# Remove unknown labels
df = df.dropna(
    subset=["label"]
)


df["label"] = df["label"].astype(int)


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates(
    subset=["message"]
)


print("\nClean dataset size:", len(df))


# ============================================================
# 6. TEXT PREPROCESSING FUNCTION
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # Keep email information
    text = re.sub(
        r"@",
        " EMAILAT ",
        text
    )

    # Replace dots
    text = re.sub(
        r"\.",
        " DOT ",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-zA-Z0-9_ ]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 7. APPLY PREPROCESSING
# ============================================================

df["clean_message"] = df["message"].apply(
    clean_text
)


print("\n====================================")
print("PREPROCESSING REPORT")
print("====================================")


print("\nBEFORE PREPROCESSING:")

for text in df["message"].head(5):

    print(
        "->",
        text
    )


print("\nAFTER PREPROCESSING:")

for text in df["clean_message"].head(5):

    print(
        "->",
        text
    )


# ============================================================
# 8. INPUT AND OUTPUT
# ============================================================

X = df["clean_message"]

y = df["label"]


# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n====================================")
print("DATASET SPLIT")
print("====================================")

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 10. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 2),

    min_df=1,

    max_df=0.95,

    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)


print("\n====================================")
print("TF-IDF")
print("====================================")

print(
    "Training matrix:",
    X_train_tfidf.shape
)

print(
    "Testing matrix:",
    X_test_tfidf.shape
)


# ============================================================
# 11. MACHINE LEARNING MODEL
# ============================================================

model = MultinomialNB()


model.fit(
    X_train_tfidf,
    y_train
)


print("\nModel training completed!")


# ============================================================
# 12. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 13. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n====================================")
print("MODEL RESULTS")
print("====================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(

        y_test,

        y_pred,

        target_names=[
            "Not Spam",
            "Spam"
        ],

        zero_division=0
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 16. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "spam_model.pkl"
)


joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


print("\n====================================")
print("MODEL FILES SAVED")
print("====================================")

print("spam_model.pkl")
print("tfidf_vectorizer.pkl")


# ============================================================
# 17. NEW EMAIL PREDICTION
# ============================================================

def predict_email(email):

    email = email.strip().lower()

    # Check whether the input is ONLY an email address
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if re.fullmatch(email_pattern, email):

        # Suspicious words commonly found in spam/phishing addresses
        suspicious_words = [
            "spam",
            "alert",
            "security",
            "verify",
            "verification",
            "winner",
            "prize",
            "reward",
            "urgent",
            "paypal-alert",
            "bank-alert",
            "account-alert"
        ]

        # Check suspicious sender address
        for word in suspicious_words:

            if word in email:

                return "SPAM", 95.0

        # Normal email address
        return "NOT SPAM", 95.0


    # ============================================
    # Normal email/message
    # ============================================

    cleaned_email = clean_text(email)

    email_vector = vectorizer.transform(
        [cleaned_email]
    )

    prediction = model.predict(
        email_vector
    )[0]

    probabilities = model.predict_proba(
        email_vector
    )[0]


    if prediction == 1:

        result = "SPAM"

        confidence = probabilities[1] * 100

    else:

        result = "NOT SPAM"

        confidence = probabilities[0] * 100


    return result, confidence
# ============================================================
# 18. TEST EMAILS
# ============================================================

print("\n====================================")
print("TEST EMAILS")
print("====================================")


test_emails = [

    "Congratulations! You won a free lottery prize!",

    "Hi, can we meet tomorrow?",

    "URGENT! Claim your free cash reward now!",

    "Please send me the project report.",

    "support-paypal-alert@secure-paypal-login.com Your account requires urgent verification.",

    "security@bank-alerts.com Your bank account has been suspended. Verify now."

]


for email in test_emails:

    result, confidence = predict_email(
        email
    )

    print("\nEmail:")

    print(email)

    print(
        "Prediction:",
        result
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )


# ============================================================
# 19. INTERACTIVE SPAM CHECKER
# ============================================================

print("\n====================================")
print("INTERACTIVE SPAM CHECKER")
print("====================================")

print("Enter an email message.")

print("Type 'exit' to stop.")


while True:

    email = input(
        "\nEnter email: "
    )


    if email.lower().strip() == "exit":

        print(
            "\nProgram stopped."
        )

        break


    if not email.strip():

        print(
            "Please enter an email."
        )

        continue


    result, confidence = predict_email(
        email
    )


    print("\n----------------------------")

    print(
        "Prediction:",
        result
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("----------------------------")
