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

if "history" not in st.session_state:
    st.session_state.history = []
if "show_all_history" not in st.session_state:
    st.session_state.show_all_history = False

def extract_file(file):
  if file.name.endswith(".txt"):
    return file.read().decode("utf-8")
  elif file.name.endswith(".docx"):
    doc = docx.Document(file)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
    return full_text
  elif file.name.endswith(".pdf"):
    doc = fitz.open(stream = file.read(), filetype = "pdf")
    text = ""
    for page in doc:
      text += page.get_text()
    return text
  else :
    st.warning("File type unknown")
    return []

def preprocess_text(text):
    return text.lower().strip()


def predict(text_list):
    try:
        vect = vectorizer.transform(text_list).toarray()
        prob = model.predict_proba(vect)
        pred = model.predict(vect)[0]  
        
        pred_index = list(model.classes_).index(pred)
        pred_confidence = prob[:, pred_index][0]
        
        readable_label = "Spam" if pred == "spam" else "Not Spam"
        
        st.markdown(f"""
        ### Prediction: **{readable_label}**
        Confidence: **{pred_confidence * 100:.2f}%**
        """)
        
        return readable_label
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
  
input_type = st.segmented_control("Pilih tipe input:", ["Text", "File"])

if input_type == "Text":
    email_text = st.text_area("Write your email here:")
    if st.button("Predict"):
        if email_text.strip() != "":
            clean_text = preprocess_text(email_text)
            hasil, confidence = predict([clean_text])
            st.markdown(f"""
            ### Prediction: **{hasil}**
            Confidence: **{confidence * 100:.2f}%**
            """)
            # Simpan ke history
            st.session_state.history.insert(0, {"text": email_text, "result": hasil, "confidence": confidence})
        else:
            st.warning("Please input text.")
else:
    file = st.file_uploader("Upload file:", type=['txt', 'docx', 'pdf'])
    if st.button("Predict"):
        if file is not None:
            full_text = extract_file(file)
            if full_text.strip() == "":
                st.warning("File kosong atau format tidak dikenali.")
            else:
                clean_text = preprocess_text(full_text)
                st.write("Preview file content:")
                st.write(full_text[:300] + "..." if len(full_text) > 300 else full_text)
                hasil, confidence = predict([clean_text])
                st.markdown(f"""
                ### Prediction: **{hasil}**
                Confidence: **{confidence * 100:.2f}%**
                """)
                # Simpan ke history
                st.session_state.history.insert(0, {"text": full_text, "result": hasil, "confidence": confidence})
        else:
            st.warning("Please upload a file.")


if st.session_state.history:
    st.subheader("History Terbaru")
    for item in st.session_state.history[:2]:
        st.write(f"**Input:** {item['text'][:100]}{'...' if len(item['text'])>100 else ''}")
        st.write(f"**Result:** {item['result']} (Confidence: {item['confidence']*100:.2f}%)")
        st.write("---")

    if len(st.session_state.history) > 2:
        if st.button("View More History"):
            for item in st.session_state.history[2:]:
                st.write(f"**Input:** {item['text'][:100]}{'...' if len(item['text'])>100 else ''}")
                st.write(f"**Result:** {item['result']} (Confidence: {item['confidence']*100:.2f}%)")
                st.write("---")


