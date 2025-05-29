import streamlit as st
import joblib
import pandas as pd
import docx
import fitz

model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title('E-mail Spam Classifier')
st.write('Input your email text or upload file')
st.write('File type accepted : txt, docx and pdf')

email_text = st.text_area("Write your email here : ")

file = st.file_uploader("Upload file here: ", type = ['txt', 'docs', 'pdf'])

def extract_file(file):
  if file.name.endswith(".txt"):
    return file.read().decode("utf-8").splitlines()
  elif file.name.endswith(".docx"):
    doc = docx.Document(file)
    return [para.text for para in doc.paragraphs if para.text.strip() != ""]
  elif file.name.endswith(".pdf"):
    doc = fitz.open(stream = file.read(), filetype = "pdf")
    text = ""
    for page in doc:
      text += page.get_text()
    return text.split("\n")
  else :
    st.warning("File type unknown")
    return []
    
def predict(text_list):
    try:
        vect = vectorizer.transform(text_list).toarray()
        prob = model.predict_proba(vect)
        spam_index = list(model.classes_).index("spam")
        spam_score = prob[:, spam_index][0]
        st.write(f"Spam score: {spam_score:.2f}")
        return ["Spam" if spam_score > 0.5 else "Not Spam"]
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
        return ["ERROR"]
      
def label_converter(label):
    if label == "ham":
        return "Not Spam"
    elif label == "spam":
        return "Spam"
    else:
        return label

if st.button("Predict"):
  if email_text:
    hasil = predict([email_text.strip()])
    st.success(f"Prediction: {label_converter(hasil[0])}")
  elif file:
    texts = extract_text_from_file(file)
    if texts:
      full_text = " ".join(texts)
      hasil = predict([full_text.strip()])
      st.success(f"Prediction: {label_converter(hasil[0])}")
      for text, pred in zip(texts, hasil):
        if text.strip():
          st.write(f"{text[:100]} -> {pred}")
else:
  st.warning("Input text or file")
  


