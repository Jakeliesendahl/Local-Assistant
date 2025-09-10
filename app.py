# app.py
import streamlit as st
from core.llm import LLMClient, LLMError

st.set_page_config(page_title="Local Assistant – Hello", page_icon="🤖")
st.title("Local Assistant – Hello World")

prompt = st.text_input("Prompt:", "Say exactly: Hello World")
go = st.button("Send")

if go:
    try:
        llm = LLMClient(model="llama3")
        with st.spinner("Thinking locally..."):
            out = llm.chat(prompt, stream=False)
        st.success("Done!")
        st.text(out)
    except LLMError as e:
        st.error(str(e))
