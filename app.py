import numpy as np
import pandas as pd
import joblib
import streamlit as st

model = joblib.load('best_model.pkl')

def prediction(date,commodity,price):
    py = model.predict(np.array([date,commodity,price]))
    return py