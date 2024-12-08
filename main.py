import streamlit as st
import pandas as pd
import os
import uuid
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile",
)

st.title("AI job offers scraper")


if st.button("Show first 3 employees"):
    with open("employees.csv") as f:
        df = pd.read_csv(f)
        st.write(df.head(3))

if st.button("Show all employees"):
    with open("employees.csv") as f:
        df = pd.read_csv(f)
        st.write(df)

with st.form(key="talk_form"):
    talk_input = st.text_input(label="Talk with AI")
    talk_button = st.form_submit_button(label="Submit")

if talk_button:
    st.write(f"{talk_input}: {llm.invoke(talk_input).content}")

if "submitted_data" not in st.session_state:
    st.session_state.submitted_data = ""

with st.form(key="url_form"):
    url_input = st.text_input(
        label="Provide me url with job effer", placeholder="https://example.com"
    )
    url_button = st.form_submit_button(label="Submit")

if url_button:
    if url_input.startswith("http://") or url_input.startswith("https://"):
        loader = WebBaseLoader(url_input)
        page_data = loader.load().pop().page_content
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the 
            following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):    
            """
        )

        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={"page_data": page_data})
        json_parser = JsonOutputParser()
        json_res = json_parser.parse(res.content)
        json_res

        st.session_state.submitted_data = json_res
    else:
        st.error("Please enter a valid URL starting with 'http://' or 'https://'")


if st.button("List names matches"):

    df = pd.read_csv("employees.csv")

    client = chromadb.PersistentClient("vectorstore")
    collection = client.get_or_create_collection(name="portfolio")

    if not collection.count():
        for _, row in df.iterrows():
            collection.add(
                documents=row["technologyStack"],
                metadatas={"lastName": row["lastName"]},
                ids=[str(uuid.uuid4())],
            )

    links = collection.query(
        query_texts=st.session_state.submitted_data[0]["skills"], n_results=2
    ).get("metadatas", [])
    st.write(links)
