import application
import report
import about
import streamlit as st

from PIL import Image

PAGES = {
    "Application": application,
    "Report":report,
    "About": about
}

img = Image.open('./assets/herbie_number.png')
st.set_page_config(page_title='herbie_vision', page_icon=img)
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()