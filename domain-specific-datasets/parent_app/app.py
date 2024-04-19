import streamlit as st

st.set_page_config("Domain Data Grower", page_icon="🧑‍🌾")

st.header("🧑‍🌾 Domain Data Grower")
st.divider()

introduction = """
## 🌱 Create a dataset seed for aligning models to a specific domain

This app helps you create a dataset seed for building diverse domain-specific datasets for aligning models.
Alignment datasets are used to fine-tune models to a specific domain or task, but as yet, there's a shortage of diverse datasets for this purpose.

## 🚜 How it works

You can create a dataset seed by defining the domain expertise, perspectives, topics, and examples for your domain-specific dataset. 
The dataset seed is then used to generate synthetic data for training a language model.

## 🗺️ The process

Define the project details, including the project name, domain, and API credentials. A dataset repository will be created, and also a personalised Streamlit app in your Hugging Face profile, with which you'll be able to define the domain expertise, perspectives, topics, and examples for your domain-specific dataset, and generate the synthetic data.

## 👩🏽‍🌾 Current Projects
WIP

"""
st.markdown(introduction)
