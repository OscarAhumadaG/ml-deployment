# ML Deployment Examples 🚀

Welcome to the **ML Deployment** repository!  
Here, I showcase how to deploy machine learning models using **Flask**, **FastAPI**, **Streamlit**, **Docker**, and **Kubernetes**.

This repository contains **three deployment projects** that demonstrate different strategies to expose ML models in production environments.

---

## 📦 Projects Overview

| Project | Framework | Deployment Tools | Description |
|:--------|:-----------|:-----------------|:------------|
| **Flask Deployment** | Flask | Docker | Deploy a classification model using a lightweight Flask API. |
| **FastAPI Deployment** | FastAPI | Docker | Serve a prediction model with FastAPI, including automatic docs (Swagger UI). |
| **Streamlit Deployment** | Streamlit | Docker | Create an interactive web app to visualize model predictions with Streamlit. |

---

## 🔥 Technologies Used

- Python 3.10
- Flask
- FastAPI
- Streamlit
- Docker
- Kubernetes (Minikube)
- Scikit-Learn
- Pandas
- Numpy

---

## 🚀 Getting Started

Clone the repository:
```bash
git clone https://github.com/OscarAhumadaG/ml-deployment.git
cd ml-deployment

Example for **Flask Deployment**:
```bash
cd flask_deployment
docker build -t flask-ml-app .
docker run -p 5000:5000 flask-ml-app
