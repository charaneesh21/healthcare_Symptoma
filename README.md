# 🩺 Symptoma – AI-Powered Symptom and Diagnosis Companion

This project builds a smart, interpretable medical assistant that predicts likely diseases from symptoms and provides natural-language explanations. The system combines machine learning (ML), semantic search, and large language models (LLMs) in a **four-stage hybrid pipeline**.

---

## ✅ Project Highlights

- **Stage 1:** ML-based disease classification using structured binary symptom vectors
- **Stage 2:** Embedding of free-text symptom descriptions using SentenceTransformers
- **Stage 3:** FAISS-based semantic retrieval of similar cases
- **Stage 4:** Explanation generation using the Mistral-7B model

---

## 🔧 Step-by-Step Pipeline

### 🔹 Step 1 – Train Random Forest Classifier
- Dataset: Top 10 balanced disease classes, ~300 samples each
- Features: Binary vector indicating presence/absence of symptoms
- Model: `RandomForestClassifier` from Scikit-learn
- Metrics:
  - **Accuracy:** 0.98
  - **Precision:** 0.98
  - **Recall:** 0.98
  - **F1-Score:** 0.98
  
  - <img width="512" alt="Screenshot 2025-05-28 at 6 38 52 PM" src="https://github.com/user-attachments/assets/5931b89b-5d0a-4cf9-975d-e9e80de5842c" />
    <img width="396" alt="Screenshot 2025-05-28 at 6 38 30 PM" src="https://github.com/user-attachments/assets/86f02967-25fd-4710-8223-3cd9c7eaf544" />


### 🔹 Step 2 – Symptom Embedding
Model: all-mpnet-base-v2 from SentenceTransformers

Converts symptom descriptions (e.g., "headache, sore throat") into 768-dimensional vectors

Enables free-text user queries


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(symptom_texts)


### 🔹 Step 3 – Semantic Retrieval using FAISS
Vector database: FAISS IndexFlatIP (cosine similarity)

Indexed ~246K symptom embeddings

Retrieval effectiveness:

~85% top-5 match accuracy

~88% top-1 match diagnostic relevance

import faiss
index = faiss.IndexFlatIP(768)
index.add(embeddings)

<img width="529" alt="Screenshot 2025-05-28 at 6 40 24 PM" src="https://github.com/user-attachments/assets/893b8aa2-8341-4357-880c-fdfb60bb54e0" />

### 🔹 Step 4 – Diagnosis Explanation via LLM (Mistral-7B)
Inputs: Top-K retrieved symptom-diagnosis pairs

Uses prompt engineering to guide Mistral-7B for response generation

Output: 1–3 sentence contextual explanation that highlights matching symptoms and justifies the prediction

Example Output:
"The reported symptoms closely match cases of Influenza. Fever, sore throat, and headache are typical, and the absence of rashes makes measles less likely."

<img width="350" alt="Screenshot 2025-05-28 at 6 42 01 PM" src="https://github.com/user-attachments/assets/38eb3737-d4c7-47e4-84a5-b713cc5cf58f" />

🧪 Evaluation Summary

| Component        | Technique                  | Performance           |
| ---------------- | -------------------------- | --------------------- |
| Classifier       | Random Forest (10 classes) | **Accuracy = 0.98**   |
| Embedding Model  | all-mpnet-base-v2          | 768-d vectors         |
| Retrieval        | FAISS (cosine similarity)  | Top-5 ≈ 85% match     |
| Explanation Gen. | Mistral-7B (prompted)      | Contextual + accurate |

🧠 Tech Stack
Scikit-learn for ML classification

SentenceTransformers for symptom embeddings

FAISS for fast semantic similarity search

Mistral-7B or compatible LLM for explanation generation

Gradio or Streamlit for interface

⚙️ Future Improvements
Add multilingual symptom support

Fine-tune generator model on domain-specific explanations

Extend dataset to include rare diseases

Introduce confidence calibration for predictions
