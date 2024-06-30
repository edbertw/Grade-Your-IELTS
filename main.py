import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv("ielts_writing_dataset.csv")
data.head()
data = data[["Task_Type", "Question", "Essay", "Overall"]]
X = np.array(data["Essay"])
y = np.array(data["Overall"])

Vectorizer = TfidfVectorizer()
X = Vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_1 = LinearRegression()
model_2 = SVR(kernel = "rbf", C = 1.0, epsilon = 0.1)
model_3 = RandomForestRegressor(n_estimators = 100, random_state = 42)

model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)

print("Linear Regression Score:",round(model_1.score(X_test, y_test) * 100,2), "%")
print("Support Vector Score:", round(model_2.score(X_test, y_test) * 100,2), "%")
print("Random Forest Score:", round(model_3.score(X_test, y_test) * 100,2), "%")

y_1 = model_1.predict(X_test)
y_2 = model_2.predict(X_test)
y_3 = model_3.predict(X_test)

print("R2 Score of Linear Regression =", round(r2_score(y_test, y_1), 2))
print("R2 Score of SVR =", round(r2_score(y_test, y_2), 2))
print("R2 Score of Random Forest =", round(r2_score(y_test, y_3), 2))

def round_to_nearest_half(num):
  return round(num * 2) /2
  
import streamlit as st
count = 0
def main():
    global count
    st.title("Grade Your IELTS!")
    st.write("Please enter your essay to start grading.")

    count += 1
    user_input = st.text_input("You:", key=f"user_input_{count}")

    if user_input:
        response = model_3.predict(Vectorizer.transform([user_input]).toarray())
        st.text_area("Your score:", value=round_to_nearest_half(response[0]), height=101, max_chars=None, key=f"chatbot_response_{count}")
        st.write("Thank you!")
        st.stop()

if __name__ == '__main__':
    main()
