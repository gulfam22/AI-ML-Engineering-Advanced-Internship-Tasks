# AI/ML Engineering – Advanced Internship Tasks  

This repository contains implementations of selected advanced tasks for the DevelopersHub Corporation internship.  

---

## ✅ Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API  

**Objective:**  
Build a reusable and production-ready machine learning pipeline for predicting customer churn.  

**Dataset:**  
Telco Churn Dataset.  

**Approach:**  
1. Implemented **data preprocessing** (scaling, encoding) using `Pipeline`.  
2. Trained models including **Logistic Regression** and **Random Forest**.  
3. Applied **GridSearchCV** for hyperparameter tuning.  
4. Exported the complete pipeline with **joblib** for deployment.  

---

## ✅ Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data  

**Objective:**  
Predict housing prices using both structured tabular data and house images.  

**Dataset:**  
- Housing Sales Dataset  
- Custom/ Public House Image Dataset  

**Approach:**  
1. Used **CNNs** (Convolutional Neural Networks) to extract features from house images.  
2. Processed **structured tabular data** (e.g., number of rooms, area, location, etc.).  
3. Combined image embeddings with tabular features.  
4. Trained a multimodal model.  
5. Evaluated performance using **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**.  

---

## ✅ Task 4: Context-Aware Chatbot Using LangChain or RAG  

**Objective:**  
Build a conversational chatbot that can remember context and retrieve external information during conversations.  

**Dataset:**  
Custom corpus (e.g., Wikipedia pages, internal documents, or any knowledge base).  

**Approach:**  
1. Implemented **LangChain** / **Retrieval-Augmented Generation (RAG)**.  
2. Added **context memory** to maintain conversational history.  
3. Built a **vectorized document store** for knowledge retrieval.  
4. Deployed chatbot with **Streamlit** for user interaction.  

---

## ✅ Task 5: Auto Tagging Support Tickets Using LLM  

**Objective:**  
Automatically tag support tickets into categories using a large language model (LLM).  

**Dataset:**  
Free-text Support Ticket Dataset.  

**Approach:**  
1. Applied **prompt engineering** for zero-shot classification.  
2. Compared **zero-shot vs fine-tuned** performance.  
3. Used **few-shot learning** to improve accuracy.  
4. Generated **top 3 most probable tags per ticket**.  

---

## 📊 Results  

- **Task 2:** Exported reusable ML pipeline with tuned models.  
- **Task 3:** Reported MAE and RMSE for multimodal model vs. baseline.  
- **Task 4:** Chatbot successfully retrieved context-aware answers with memory + vector store.  
- **Task 5:** Achieved higher accuracy with few-shot/fine-tuned methods compared to zero-shot.  

---

## ⚙️ How to Run  

1. Clone this repo:  
   ```bash
   git clone https://github.com/your-username/internship-tasks.git
   cd internship-tasks
