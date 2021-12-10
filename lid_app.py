import streamlit as st
from annotated_text import annotated_text
from pathlib import Path
from LanguageIdentifier import LanguageIdentifier
import pandas as pd
import altair as alt
from colour import Color
import random


def color_text(output: dict):
    tokens = output['tokens']
    preds = output['predictions']
    return [(tok, pred, lang_color_dict[pred]['color']) for tok, pred in zip(tokens, preds)]


def get_example_sents(filepath: str) -> list[str]:
    sentences = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            sentences.append(line)
    return sentences


def gen_colors(num_tokens: int):
    start = Color('#b2eab4')
    colors = list(start.range_to(Color("#007e5f"), num_tokens))
    color_strings = [color.hex for color in colors]
    return color_strings


def make_conf_chart(data_output: dict):
    color_range = gen_colors(len(data_output['tokens']))
    sorted_conf = sorted(list(data_output['confidence']))

    source = pd.DataFrame(data_output)
    chart = alt.Chart(source).mark_bar().encode(
        x=alt.X('tokens:N', sort=data_output['tokens']),
        y=alt.Y('confidence:Q', scale=alt.Scale(domain=(0.0, 1.0), clamp=True)),
        color=alt.Color('confidence:O',
                        scale=alt.Scale(domain=sorted_conf,
                                        range=color_range))
    ).properties(width=1200).configure_axisX(labelAngle=0)
    return chart


def make_rank_chart(data_output: dict, rank: dict[str, list]):
    sent_len = len(data_output['tokens'])
    tokens = [t for tok_chunk in [[tok] * 6 for tok in data_output['tokens']] for t in tok_chunk]
    lang_tags = [tag for word in [list(rank.keys()) * sent_len] for tag in word]
    confidence = [word_confs[i] for i in range(sent_len) for lang, word_confs in rank.items()]
    color_pairs = [(lang, val['color']) for lang, val in lang_color_dict.items()]

    assert len(tokens) == len(lang_tags) == len(confidence)
    source = pd.DataFrame({'tokens': tokens,
                           'lang': lang_tags,
                           'confidence': confidence})

    chart = alt.Chart(source).mark_bar(opacity=0.75).encode(
        x=alt.X('tokens:N', sort=tokens),
        y=alt.Y('confidence:Q', scale=alt.Scale(domain=(0, 1.0), clamp=True)),
        color=alt.Color('lang:N', scale=alt.Scale(domain=[it[0] for it in color_pairs],
                                                  range=[it[1] for it in color_pairs]))
    ).properties(width=1200, height=200).configure_axisX(labelAngle=0)
    return chart


lang_color_dict = {'bn': {'name': 'Bangla', 'color': '#ff8da9'},
                   'en': {'name': 'English', 'color': '#78f6ff'},
                   'univ': {'name': 'Universal', 'color': '#cf9cff'},
                   'ne': {'name': 'Named Entity', 'color': '#ffdd69'},
                   'acro': {'name': 'Acronym', 'color': '#c0ff6f'}}

LID = LanguageIdentifier(Path("trained_models/trained_LID_model.pth"))


# ---------------------------------------------------------------------------------------------------#
# Page layout
# Page expands to full width
def write_web_app():
    st.set_page_config(page_title='Bangla-English Code-mixing Identifier', layout='wide')
    st.write("""
    # Bangla-English Code-mixing Identifier
    This app predicts the **language** for each token in code-mixed Bangla-English social media data.
    Uses a single-layer BiLSTM with sub-word embeddings generated using Google SentencePiece. See all the code [here](https://github.com/aparnadutta/nlp-systems-final)!
    """)

    annotated_text(*[(d['name'], '', d['color']) for code, d in lang_color_dict.items()])
    st.text("")
    all_sents = get_example_sents('example_sentences.txt')
    rand_sent = random.choice(all_sents)

    gen_button = st.checkbox(label='Generate test sentences')

    form = st.form(key='my_form')
    if gen_button:
        sentence = form.text_input(label='Type in some words', value=rand_sent)
        submit_button = form.form_submit_button(label='Generate')

    else:
        sentence = form.text_input(label='Type in some words')
        submit_button = form.form_submit_button(label='Submit')

    if submit_button:
        text, output = LID.predict(sentence)
        rank = LID.rank(sentence)

        st.subheader('Predicted Labels')
        annotated_text(*color_text(output))
        # st.text("")
        st.subheader('Word-level Model Confidence')
        # st.altair_chart(make_conf_chart(output))
        st.altair_chart(make_rank_chart(output, rank))


if __name__ == '__main__':
    write_web_app()
