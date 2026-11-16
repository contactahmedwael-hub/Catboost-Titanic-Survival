# Titanic Survival Predictor (CatBoost + Streamlit)

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder's contents (`app.py`, `requirements.txt`, `train.csv`) to a GitHub repository (files at the repo root).
2. Go to https://share.streamlit.io, click **Create app**, pick the repo, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python 3.11 or 3.12, then **Deploy**.

The model is trained on startup (a few seconds) and cached, so no model file is needed.
