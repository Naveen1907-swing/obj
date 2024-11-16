from app import main
import streamlit as st

def handler(request, response):
    return main()
