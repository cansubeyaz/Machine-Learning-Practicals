from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

'''
This code uses a small dataset with text samples labeled as positive or negative. It then tokenizes the text, converts it 
to a bag-of-words representation using CountVectorizer, and trains a Multinomial Naive Bayes classifier using MultinomialNB. 
Finally, it evaluates the classifier's accuracy on a test set.
'''

# Sample data
corpus = [
    ("I love this sandwich.", "positive"),
    ("This is an amazing place!", "positive"),
    ("I feel very good about these beers.", "positive"),
    ("This is my best work.", "positive"),
    ("What an awesome view", "positive"),
    ("I do not like this restaurant", "negative"),
    ("I am tired of this stuff.", "negative"),
    ("I can't deal with this", "negative"),
    ("He is my sworn enemy!", "negative"),
    ("My boss is horrible.", "negative")
]

# Separate data and labels
X, y = zip(*corpus)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into feature vectors
'''
This step converts text into a bag-of-words representation, which counts the frequency of each word in the text.'''
vectorizer = CountVectorizer() ## bag-of-words representation
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Predict on test data
y_pred = clf.predict(X_test_counts)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
