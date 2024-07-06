import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import streamlit as st
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from keras.metrics import MeanAbsoluteError
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

data = pd.read_csv("ielts_writing_dataset.csv")
data.head()
data = data[(data['Overall'].value_counts()) > 1]

X = data["Essay"]
y = data["Overall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y, random_state=42)
X_train = bert_tokenizer(list(X_train), padding=True, truncation=True, return_tensors='tf', max_length = 512)['input_ids']
X_test = bert_tokenizer(list(X_test), padding=True, truncation=True, return_tensors='tf', max_length = 512)['input_ids']
'''
reg = keras.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='linear')
])
'''
reg = Sequential()
reg.add(Flatten())
reg.add(Dense(64, activation='relu'))
reg.add(Dropout(0.3))
reg.add(Dense(32, activation='relu'))
reg.add(Dropout(0.3))
reg.add(Dense(1, activation='linear'))

class TFBertModelWrapper(Layer):
    def __init__(self, **kwargs):
        super(TFBertModelWrapper, self).__init__(**kwargs)
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        return outputs[0]

input_ids = keras.layers.Input(shape=(512,), dtype=tf.int32)

output = TFBertModelWrapper()(input_ids)
pooling = output.pooler_output
output_ids = reg(pooling)

model = keras.Model(inputs = input_ids, outputs = output_ids)
for layer in bert_model.layers:
    layer.trainable = False

model.compile(optimizer = "adam",
             loss = "mean_squared_error",
             metrics = ["mean_absolute_error"])

early_stopping = keras.callbacks.EarlyStopping(
    patience=8,
    min_delta=0,
    monitor = "val_loss",
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=50,
    callbacks=[early_stopping],
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="val loss")
history_frame.loc[:, ['mean_absolute_error', 'val_mean_absolute_error']].plot()

y_pred = model.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

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

# Use Best Model (Fine-Tuned BERT)
count = 0
def main():
    global count
    st.title("Grade Your IELTS!")
    st.write("Please enter your essay to start grading.")

    count += 1
    user_input = st.text_input("You:", key=f"user_input_{count}")

    if user_input:
        new_input_ids = bert_tokenizer(user_input, padding=True, truncation=True, return_tensors='tf', max_length=512)['input_ids']
        response = model.predict(new_input_ids)
        st.text_area("Your score:", value=round_to_nearest_half(response), height=101, max_chars=None, key=f"chatbot_response_{count}")
        st.write("Thank you!")
        st.stop()

if __name__ == '__main__':
    main()
