FROM python:3.11-slim

WORKDIR /app

# requirements 먼저 (캐시 활용)
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY data/graph_output/ ./data/graph_output/

# __init__.py 생성 (패키지 인식용)
RUN touch src/__init__.py src/api/__init__.py src/api/DK/__init__.py \
    src/core/__init__.py src/generators/__init__.py src/rapo/__init__.py

# 환경변수
ENV PORT=8888
ENV PYTHONUNBUFFERED=1

# 실행
CMD ["uvicorn", "src.api.DK.main:app", "--host", "0.0.0.0", "--port", "8888"]
