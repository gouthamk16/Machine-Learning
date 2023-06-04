
## Spam Filter Model - 21BAI1007
## Solo

# In this code a UI interface is made to take user input and output whether the message is spam or not.
# Sir, please make sure the csv file is in the same directory. I am importing the file from local drive

### 1. Importing Necessary Libraries and Modules


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import preprocessing 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

### 2. Importing and loading the dataset"""

df = pd.read_csv("spam.csv", encoding='latin1')
df.head()

### 3. Data Preprocessing

# Removing the unwanted columns generated as a result of encoding
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', "Unnamed: 4"], inplace=True)

# Renaming the column 'v1' to 'category' and 'v2' to 'message'
df.rename(columns = {'v1':'Category'}, inplace = True)
df.rename(columns = {'v2':'Message'}, inplace = True)
df.head()

# Inspecting the data
df.groupby('Category').describe()

# Encoding 'category' according to spam or not spam.
# 1 if spam, 0 if not
df['spam'] = df['Category'].apply(lambda x:1 if x=='spam' else 0)
df.head()

# Splitting the data into train and test
x = df.Message
y = df.spam
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x_train.describe(), x_test.describe()

y_train.describe(), y_test.describe()

# Counting the frequecy of each word using the count vectorizer
# Finding the word count and storing the data as a numerical matrix
count = CountVectorizer()
x_train_count = count.fit_transform(x_train.values)
x_train_count.toarray()

### 3. Training the model

# Declaring and fitting our data to the naive bayes model
classifier = MultinomialNB()
classifier.fit(x_train_count, y_train)

# Pretest
email_ham = ["You won 1000$. Click the link below"]
email_ham_count = count.transform(email_ham)
classifier.predict(email_ham_count)

### 4. Testing the model

# Count vectorizing the test dataset
x_test_count = count.transform(x_test)
# Testing the accuracy
classifier.score(x_test_count, y_test)*100

### 5. User Inputs and Final Output

# user_message = input()
# inp = [user_message]
# message_count = count.transform(inp)
# print(classifier.predict(message_count))

import tkinter as tk
from tkinter import messagebox

# Function to handle the button click event
def classify_message():
    user_message = message_input.get()
    inp = [user_message]
    message_count = count.transform(inp)
    result = classifier.predict(message_count)
    if result[0]==1:
        messagebox.showinfo("Spam", "Message is spam")
    else:
        messagebox.showinfo("Ham", "Message is genuine")

    #messagebox.showinfo("Spam Filter Result", result[0])

# Creating the main window
window = tk.Tk()
window.title("Spam Filter Model")
window.geometry("400x200")

# Label for the message input
message_label = tk.Label(window, text="Enter your message:")
message_label.pack()

# Entry box for the message input
message_input = tk.Entry(window, width=50)
message_input.pack()

# Button to classify the message
classify_button = tk.Button(window, text="Classify", command=classify_message)
classify_button.pack()

# Running the main loop
window.mainloop()
