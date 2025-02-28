import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Read the CSV file
spam = pd.read_csv('/Users/sidkumar/Documents/Portfolio Freelance/MailSpam/spam.csv')


# Use the correct column name for the email text
z = spam['v2']
y = spam['v1']

# Split the data into training and testing sets
z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2)

# Vectorize the text data
cv = CountVectorizer()
features = cv.fit_transform(z_train)

# Train the SVM model
model = svm.SVC()
model.fit(features, y_train)

# Transform the test features
features_test = cv.transform(z_test)

# Print the accuracy up to two decimal places
accuracy = model.score(features_test, y_test) * 100
print(f'Accuracy: {accuracy:.2f}%')
