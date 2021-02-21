from streamlit_app import application
from streamlit_app import report
from streamlit_app import about
import streamlit as st

PAGES = {
    "Application": application,
    "Report":report,
    "About": about
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()