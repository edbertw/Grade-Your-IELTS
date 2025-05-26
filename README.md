# Grade Your IELTS Writing! 🚀

Welcome to **Grade Your IELTS**, an open-source project that leverages advanced Natural Language Processing (NLP) models to automatically assess and grade IELTS Writing responses. This tool is designed to help students, educators, and researchers evaluate IELTS essays with instant, AI-powered feedback.

---

## ✨ Features

- **Automatic Grading of IELTS Writing Tasks**
- **Multiple ML and Transformer Models** evaluated (Linear Regression, SVR, Random Forest, BERT variants, etc.)
- **State-of-the-art Fine-Tuned DeBERTa V3** Deployed for Best Performance
- **Interactive Web UI** powered by Dash for easy experimentation
- **Transparent Metrics** for all models and approaches

---

## 🚦 Model Performance

| Model                           | R<sup>2</sup> Score | MAE      |
|----------------------------------|:------------------:|:--------:|
| Linear Regression                | 0.16               |    -     |
| SVR                              | 0.23               |    -     |
| Random Forest                    | 0.25               |    -     |
| BERT + Untrained Head            | 0.3063             | 0.694    |
| Fine-tuned DistilBERT            | 0.3904             | 0.6386   |
| Fine-tuned RoBERTa               | 0.2414             | 0.717    |
| Fine-tuned BERT                  | 0.468              | 0.6091   |
| **Fine-tuned DeBERTa V3**        | **0.5057**         | **0.5759** |
| Fine-tuned ALBERT V2             | 0.0683             | 0.76443  |

> **Note:** Fine-Tuned DeBERTa V3 achieves the highest performance and is the model deployed in production.

---

## 🖥️ Live Demo

The app is deployed as an interactive web tool using [Dash](https://dash.plotly.com/). Users can input IELTS Writing essays and receive instant scores and feedback.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/edbertw/Grade-Your-IELTS.git
cd Grade-Your-IELTS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download/Prepare Model Weights

Download the fine-tuned DeBERTa V3 weights and place them in the appropriate directory as specified in the code/configuration.

### 4. Run the Dash App

```bash
python app.py
```

The app will run locally at `http://127.0.0.1:8050/` by default.

---

## 🛠️ Project Structure

```
Grade-Your-IELTS/
├── app.py                # Dash web app entry point
├── model/                # ML & Transformer model code and weights
├── data/                 # Data processing scripts and datasets
├── requirements.txt      # Python dependencies
├── utils/                # Helper functions
└── README.md
```

---

## 📊 Methodology

- **Data Preprocessing:** Essay cleaning, tokenization, and feature engineering.
- **Model Training:** Traditional ML (Linear Regression, SVR, Random Forest) and Transformer approaches (BERT, DistilBERT, RoBERTa, DeBERTa, ALBERT).
- **Evaluation Metrics:** R<sup>2</sup> score for regression quality, Mean Absolute Error (MAE) for grading accuracy.
- **Deployment:** Best model (Fine-tuned DeBERTa V3) is wrapped in a Dash app for real-world use.

---

## 🤖 Example Usage

1. Input your IELTS Writing Task 2 essay into the provided field.
2. Click "Grade" to receive an instant score and feedback.
3. Review the AI-generated comments to improve your writing.

---

## 🤝 Contributions

Contributions, suggestions, and feature requests are welcome! Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/) for pre-trained models
- [Dash by Plotly](https://dash.plotly.com/) for the web framework
- The open-source community for sample datasets and inspiration

---

**Empowering IELTS learners and teachers with AI!**
