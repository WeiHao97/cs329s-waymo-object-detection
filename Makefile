.PHONY: run-streamlit run-streamlit-container gcloud-deploy-streamlit

run-streamlit:
	@streamlit run herbie_vision/streamlit_app/app.py --server.port=8080 --server.address=0.0.0.0

run-streamlit-container:
	@docker build herbie_vision/streamlit_app/. -t herbie_vision
	@docker run -p 8080:8080 herbie_vision

gcloud-deploy-streamlit:
	@gcloud app deploy herbie_vision/streamlit_app/app.yaml

