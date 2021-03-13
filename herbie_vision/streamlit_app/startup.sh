#!/bin/bash
gcsfuse --implicit-dirs app-segments /home/data
streamlit run app.py --server.port=8080 --server.address=0.0.0.0