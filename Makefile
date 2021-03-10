.PHONY: run-streamlit run-streamlit-container-locally gcloud-deploy-streamlit \
		gcloud-deploy-web-application gcloud-tear-down-prediction-web-application \
		gcloud-run-waymo-data-processing gcloud-train-job \
		gcloud-train-sweep



run-streamlit-container-locally:
	@docker build herbie_vision/streamlit_app/. -t herbie_vision
	@docker run -p 8080:8080 herbie_vision


gcloud-deploy-web-application:
	@gcloud container clusters create thor --flags-file ./config/model_serving/cluster_config.yaml
	@gcloud container clusters get-credentials thor --zone us-central1-c --project waymo-2d-object-detection
	@kubectl create secret docker-registry mycred \
			 --docker-server=registry.hub.docker.com/peterdavidfagan/cs329s-prediction --docker-username=peterdavidfagan \
			 --docker-password=<docker-password>  \
			 --docker-email=peterdavidfagan@gmail.com
	@kubectl apply -f ./config/model_serving/basic_pod.yaml
	@echo 'Wating for pods to be created...'
	@sleep 240
	@kubectl expose pod cs329s-prediction --type=LoadBalancer --port=80 --target-port 5000
	@echo 'Wating for external ip to be initialized...'
	@sleep 240
	@sed -i .bak "s|.*REST_API.*|ENV REST_API=\"http://$$(kubectl get services cs329s-prediction -o json | jq '.status.loadBalancer.ingress | to_entries | .[0].value.ip' | tr -d '"')/predict\"|" ./herbie_vision/streamlit_app/Dockerfile
	@gcloud app deploy --quiet herbie_vision/streamlit_app/app.yaml


gcloud-tear-down-prediction-web-application:
	@gcloud container clusters delete thor --zone us-central1-c --quiet # automate zone flag in future
	@gcloud app services delete web-application --quiet


gcloud-waymo-data-processing:
	@gcloud compute instances create-with-container thor-train \
	 --machine-type e2-standard-8 --boot-disk-size 200 \
	 --container-image gcr.io/waymo-2d-object-detection/dataprocessing_train

	@gcloud compute instances create-with-container thor-test \
	--machine-type e2-standard-8 --boot-disk-size 200 \
	--container-image gcr.io/waymo-2d-object-detection/dataprocessing_test

	@gcloud compute instances create-with-container thor-validation \
	--machine-type e2-standard-8 --boot-disk-size 200 \
	--container-image gcr.io/waymo-2d-object-detection/dataprocessing_validation

gcloud-train-job:
	@gcloud compute instances create pytorch-cpu \
	  --zone=us-central1-c \
	  --image-family=pytorch-latest-cpu \
	  --image-project=deeplearning-platform-release \
	  --machine-type e2-standard-8 \
	  --boot-disk-size 200 \
	  --metadata-from-file startup-script=mount.sh
	@sleep 120
	@gcloud compute scp --recurse <path_to_model_training_script_and_config> pytorch-cpu:/tmp
	@gcloud compute ssh pytorch-cpu --command "sudo mv /tmp/* /home/waymo"
	@gcloud compute ssh pytorch-cpu --command "pip3 install --upgrade pip setuptools wheel"
	@gcloud compute ssh pytorch-cpu --command "cd /home/waymo && pip3 install -r requirements.txt"
	@gcloud compute ssh pytorch-cpu --command "export WANDB_API_KEY=<wandb-api-key> && \
											   cd /home/waymo && \
											   python3 train.py 'gcp_credentials.yaml' 'train.yaml'"
	@gcloud compute instances delete pytorch-cpu --quiet


gcloud-train-sweep:
	@gcloud container clusters create thor --flags-file ./config/model_training/cluster_config.yaml
	@gcloud container clusters get-credentials thor --zone us-central1-c --project waymo-2d-object-detection
	@kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
	@kubectl apply -f ./config/model_training/wandb_kubernetes.yaml
	@kubectl create -f ./config/model_training/persistent_volume.yaml
	@kubectl create -f ./config/model_training/persistent_volume_claim.yaml
	@kubectl apply -f ./config/model_training/train_deployment.yaml
	@gcloud container clusters delete thor --zone us-central1-c --quiet # automate zone flag in future
