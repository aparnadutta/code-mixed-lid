import streamlit as st
from annotated_text import annotated_text
from pathlib import Path
from LanguageIdentifier import LanguageIdentifier
import pandas as pd
import random

# ---------------------------------#
# Page layout
# Page expands to full width
st.write()
st.set_page_config(page_title='Bangla-English Code-mixing Identifier', layout='wide')
st.write("""
# Bangla-English Code-mixing Identifier
This app predicts the **language** for each token in code-mixed Bangla-English social media data.
Uses a single-layer BiLSTM with sub-word embeddings generated using Google SentencePiece. See all the code [here](https://github.com/aparnadutta/nlp-systems-final)!

""")

lid = LanguageIdentifier(Path("trained_models/trained_LID_model.pth"))

colors = {'bn': '#d2a8ff', 'en': '#c2ff78', 'univ': '#78f6ff', 'ne': '#ff78a3'}
color_labels = {'Bangla': 'bn', 'English': 'en', "Universal": 'univ', "Named Entity": 'ne'}
color_tuples = [(lang, '', colors[code]) for lang, code in color_labels.items()]


def color_text(output: dict) -> annotated_text:
    tokens = output['tokens']
    preds = output['predictions']
    return [(tok, pred, colors[pred]) for tok, pred in zip(tokens, preds)]


def get_ex_sents(filepath: str) -> list[str]:
    sentences = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            sentences.append(line)
    return sentences


all_sents = get_ex_sents('example_sentences.txt')

annotated_text(*color_tuples)
st.text("")
gen_button = st.checkbox(label='Generate sentence')
st.text("")

form = st.form(key='my_form')
if gen_button:
    sentence = form.text_input(label='Type in some words', value=random.choice(all_sents))
else:
    sentence = form.text_input(label='Type in some words')

submit_button = form.form_submit_button(label='Submit')

if submit_button:
    text, output = lid.predict(sentence)
    st.subheader('Predicted Labels')
    st.text('\n\n\n\n')
    annotated_text(*color_text(output))
    st.subheader('Word-level Model Confidence')
    conf_dict = {i: conf for i, conf in enumerate(output['confidence'])}
    chart_data = pd.Series(conf_dict)
    st.bar_chart(chart_data)
