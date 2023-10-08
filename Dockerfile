FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./app /app

COPY ./models /models

COPY ./data /data

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app", "--bind", "0.0.0.0:80", "--timeout", "300"]