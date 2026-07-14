FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11-slim

ARG SERVICE_CONFIG=/app/servicer/config/api_service_hanwam_docker.yml

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_SERVICE_CONFIG=${SERVICE_CONFIG}

WORKDIR /app

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir \
        "fastapi>=0.111,<1.0" \
        "pydantic>=2,<3" \
        "uvicorn[standard]>=0.30,<1.0" \
        numpy \
        pandas \
        pyyaml

COPY control ./control
COPY data ./data
COPY servicer ./servicer

RUN mkdir -p /app/outputs/servicer

VOLUME ["/app/outputs/servicer"]

EXPOSE 8080

CMD ["sh", "-c", "python -m servicer.api_servicer.main --config ${API_SERVICE_CONFIG}"]
