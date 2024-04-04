FROM tensorflow/serving
EXPOSE 8601
RUN tensorflow_model_server --rest_api_port=8601 --model_name=model --model_base_path=models/