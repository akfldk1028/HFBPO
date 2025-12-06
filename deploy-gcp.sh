#!/bin/bash
# HFBPO Cloud Run 배포 스크립트

set -e

# 설정
PROJECT_ID="${GCP_PROJECT:-dkdk-474008}"
SERVICE_NAME="hfbpo-api"
REGION="${GCP_REGION:-asia-northeast3}"
MEMORY="${MEMORY:-1Gi}"
CPU="${CPU:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"

echo "=== HFBPO Cloud Run 배포 시작 ==="
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"

# 1. 프로젝트 루트로 이동
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# 2. data/graph_output 확인
if [ ! -d "data/graph_output" ]; then
    echo "❌ data/graph_output 폴더가 없습니다!"
    exit 1
fi
echo "✅ data/graph_output 확인됨"

# 3. OPENAI_API_KEY Secret 확인/생성
echo ""
echo "=== Secret Manager 확인 ==="
if gcloud secrets describe OPENAI_API_KEY --project=$PROJECT_ID >/dev/null 2>&1; then
    echo "✅ OPENAI_API_KEY secret 존재함"
else
    echo "⚠️ OPENAI_API_KEY secret 없음 - 생성 필요"
    echo ""
    echo "다음 명령어로 생성하세요:"
    echo "  echo -n 'sk-proj-YOUR_KEY' | gcloud secrets create OPENAI_API_KEY --data-file=- --project=$PROJECT_ID"
    echo ""
    read -p "Secret을 생성했으면 Enter를 누르세요..."
fi

# 4. Cloud Build로 배포
echo ""
echo "=== Cloud Build 시작 ==="

gcloud builds submit \
    --project=$PROJECT_ID \
    --config=deploy/cloudbuild.yaml \
    --substitutions=_SERVICE_NAME=$SERVICE_NAME,_REGION=$REGION,_MEMORY=$MEMORY,_CPU=$CPU,_MAX_INSTANCES=$MAX_INSTANCES

echo ""
echo "=== 배포 완료 ==="

# 5. 서비스 URL 확인
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --project=$PROJECT_ID \
    --region=$REGION \
    --format="value(status.url)" 2>/dev/null || echo "")

if [ -n "$SERVICE_URL" ]; then
    echo ""
    echo "🚀 Service URL: $SERVICE_URL"
    echo ""
    echo "테스트:"
    echo "  curl $SERVICE_URL/"
    echo "  curl -X POST $SERVICE_URL/generate -H 'Content-Type: application/json' -d '{\"topic\": \"한강 야경\"}'"
else
    echo "⚠️ 서비스 URL을 가져올 수 없습니다. Cloud Console에서 확인하세요."
fi
