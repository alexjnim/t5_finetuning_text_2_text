FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

COPY resources/inference resources/inference
COPY seldon seldon

ENV PYTHONPATH "${PYTHONPATH}:/app"

EXPOSE 5000
EXPOSE 9000

CMD exec seldon-core-microservice seldon.SummarisationModel.T5SUM --service-type MODEL --persistence 0