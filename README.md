# Grade IELTS with NLP and Machine Learning (Deep Learning)
## About
This algorithm is trained on a dataset consisting of Essays and Overall IELTS Score for writing. It employs a TfidfVectorizer text feature extraction algorithm to convert text to numerical vector representation (Tokenization) which can be used as features for the machine learning algorithm. I tested the feature X and target y on three separate powerful Regression algorithms as well as a fine-tuned BERT pre-trained model to see which performs best. The Fine-Tuned BERT Model performs the best, achieving an RMSE of ```0.81```. Note that due to the nature of this task, it is difficult to predict the score accurately with ML algorithms. I then deployed this algorithm to a web application using the streamlit library, so that users may input their essays in an interactive interface and environment and obtain their deserving grades with the standard IELTS grading points.
## Run Instructions
On the command line, use the command ```python3 main.py``` ,then use the command ```streamlit run main.py``` to start grading your IELTS Essay.
## Results & Method
```
Linear Regression Score: 16.19 %
Support Vector Score: 22.57 %
Random Forest Score: 25.16 %
R2 Score of Linear Regression = 0.16
R2 Score of SVR = 0.23
R2 Score of Random Forest = 0.25
```
As seen, using standard Machine Learning algorithms have proven to be less effective. However, when we employed the fine-tuned BERT model with early stopping, a relatively high number of training epochs and other parameters, it achieved an RMSE of ```0.81```, which when comparing with the previous ML models, is much more precise. 

We Fine Tuned the BERT model by feeding a Keras Object (Input Layer) to the pre-trained BERT model and obtaining its pooling output. We used this as an input to the self-designed neural network which outputs a single value using the linear activation function (Network Design = (64 (RelU),32 (RelU) ,1 (linear)) and some dropout layers to prevent overfitting and increase generalization to new data. We then created the final model and made it so that the layers from BERT will be unchanged and untrainable during the model fitting and training process.

Same as before (during training and fitting), for any new text (essays) grade that is to be predicted by the model, it is first tokenized with the BertTokenizer function from the transformers library.

Implementing these methodologies, the model has proved its importance and competence in predicting IELTS Writing grades using the IELTS Standard Grading system / points.
