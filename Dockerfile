FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 先复制依赖声明，最大化利用构建缓存层
COPY pyproject.toml ./
COPY README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

# 运行时数据目录（向量索引、上传缓存）
RUN mkdir -p /app/data

EXPOSE 8000 8501

# 默认启动 FastAPI 服务；如需 WebUI：docker compose run webui
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
