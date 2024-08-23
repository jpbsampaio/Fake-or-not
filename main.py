import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string


df = pd.read_csv("pre-processed.csv")

dados = df.sample(frac=1, random_state=42).reset_index(drop=True)

def formata(texto):
    texto = texto.lower()
    texto = re.sub(r'\[.*?\]', '', texto)
    texto = re.sub(r"\\W"," ",texto) 
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r'<.*?>+', '', texto)
    texto = re.sub(r'[%s]' % re.escape(string.punctuation), '', texto)
    texto = re.sub(r'\n', '', texto)
    texto = re.sub(r'\w*\d\w*', '', texto)    
    return texto


dados["preprocessed_news"] = dados["preprocessed_news"].apply(formata)

x = df["preprocessed_news"]
y = df["label"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25)

vetorizacao = TfidfVectorizer()
xv_treino = vetorizacao.fit_transform(x_treino)
xv_teste = vetorizacao.transform(x_teste)

LR = LogisticRegression()
LR.fit(xv_treino,y_treino)

pred_lr=LR.predict(xv_teste)


DT = DecisionTreeClassifier()
DT.fit(xv_treino, y_treino)

pred_dt = DT.predict(xv_teste)
DT.score(xv_teste, y_teste)


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_treino, y_treino)

pred_gbc = GBC.predict(xv_teste)

GBC.score(xv_teste, y_teste)

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_treino, y_treino)

pred_rfc = RFC.predict(xv_teste)
RFC.score(xv_teste, y_teste)

def output_lable(n):
    if n == "fake":
        return "Fake News"
    elif n == "true":
        return "não é fake News"


def teste(news):
    testing_news = {"label":[news]}
    new_def_teste = pd.DataFrame(testing_news)
    new_def_teste["label"] = new_def_teste["label"].apply(formata) 
    new_x_teste = new_def_teste["label"]
    new_xv_teste = vetorizacao.transform(new_x_teste)
    pred_LR = LR.predict(new_xv_teste)
    pred_DT = DT.predict(new_xv_teste)
    pred_GBC = GBC.predict(new_xv_teste)
    pred_RFC = RFC.predict(new_xv_teste)

    prob_LR = LR.predict_proba(new_xv_teste)
    prob_DT = DT.predict_proba(new_xv_teste)
    prob_GBC = GBC.predict_proba(new_xv_teste)
    prob_RFC = RFC.predict_proba(new_xv_teste)

    print(f"Predição LR: {output_lable(pred_LR[0])} Com probabilidade de: {prob_LR[0]}")
    print(f"Predição DT: {output_lable(pred_DT[0])} Com probabilidade de: {prob_DT[0]}")
    print(f"Predição GBC: {output_lable(pred_GBC[0])} Com probabilidade de: {prob_GBC[0]}")
    print(f"Predição RFC: {output_lable(pred_RFC[0])} Com probabilidade de: {prob_RFC[0]}")

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),                                                                                                       output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


noticia = str(input())
teste(noticia)