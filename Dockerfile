FROM python:3.8 AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.8-slim
RUN ls -lh .
WORKDIR /code
COPY --from=builder /root/.local /root/.local
COPY / /code
ENV PATH=/root/.local:$PATH:/code:/root/.local/bin
RUN ls -lh .
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]
