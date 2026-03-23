import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Create dataset
data = {
    "text": [
        "Payment not processed",
        "Unable to login",
        "App is crashing",
        "Need help with account",
        "Refund not received",
        "Website is slow",
        "Password reset issue",
        "Order not delivered",
        "General inquiry about product",
        "Technical error in app",
        "Billing issue with invoice",
        "Login error problem"
    ],
    "category": [
        "Billing",
        "Technical",
        "Technical",
        "General",
        "Billing",
        "Technical",
        "Technical",
        "Billing",
        "General",
        "Technical",
        "Billing",
        "Technical"
    ]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# Step 2: Convert text to numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['category']

# Step 3: Split data (train & test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Test model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Step 6: Predict new ticket
test_ticket = ["App not working properly"]
test_vector = vectorizer.transform(test_ticket)

prediction = model.predict(test_vector)
print("\nPredicted Category:", prediction[0])

# Step 7: Assign priority
if prediction[0] == "Technical":
    print("Priority: High")
elif prediction[0] == "Billing":
    print("Priority: Medium")
else:
    print("Priority: Low")
