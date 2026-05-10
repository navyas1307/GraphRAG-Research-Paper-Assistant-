#!/bin/bash
# ─────────────────────────────────────────────────────────────
# GraphRAG Research Paper Assistant — Start Script
# ─────────────────────────────────────────────────────────────
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════╗"
echo "║   GraphRAG Research Paper Assistant          ║"
echo "║   Startup Script                             ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: Check prerequisites ──────────────────────────────
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

command -v docker >/dev/null 2>&1 || { echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ Python 3 not found.${NC}"; exit 1; }
command -v node >/dev/null 2>&1 || { echo -e "${RED}❌ Node.js not found.${NC}"; exit 1; }
echo -e "  ✅ Docker, Python, Node.js found"

# ── Step 2: Start Docker services ────────────────────────────
echo -e "${YELLOW}[2/5] Starting database services (PostgreSQL, Qdrant, Neo4j)...${NC}"
docker-compose up -d

echo "  Waiting for services to be healthy..."
sleep 8

# Check health
docker ps --filter "name=research_" --format "  ✅ {{.Names}} — {{.Status}}"

# ── Step 3: Check .env ────────────────────────────────────────
echo -e "${YELLOW}[3/5] Checking backend configuration...${NC}"
if [ ! -f "backend/.env" ]; then
  echo -e "${RED}❌ backend/.env not found!${NC}"
  exit 1
fi

if grep -q "your-gemini-api-key-here" backend/.env; then
  echo -e "${YELLOW}  ⚠️  GEMINI_API_KEY not set in backend/.env${NC}"
  echo -e "     AI summaries will use keyword fallback."
else
  echo -e "  ✅ .env configured"
fi

# ── Step 4: Backend ───────────────────────────────────────────
echo -e "${YELLOW}[4/5] Starting backend (FastAPI on :8000)...${NC}"
cd backend

if [ ! -d "venv" ]; then
  echo "  Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt -q

mkdir -p uploads

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "  ✅ Backend started (PID $BACKEND_PID)"
cd ..

sleep 3

# ── Step 5: Frontend ──────────────────────────────────────────
echo -e "${YELLOW}[5/5] Starting frontend (Next.js on :3000)...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
  echo "  Installing npm dependencies..."
  npm install -q
fi

npm run dev &
FRONTEND_PID=$!
echo "  ✅ Frontend started (PID $FRONTEND_PID)"
cd ..

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗"
echo "║   🚀 All services running!                   ║"
echo "║                                              ║"
echo "║   Frontend:  http://localhost:3000           ║"
echo "║   Backend:   http://localhost:8000           ║"
echo "║   API Docs:  http://localhost:8000/docs      ║"
echo "║   Qdrant:    http://localhost:6333/dashboard ║"
echo "║   Neo4j:     http://localhost:7474           ║"
echo "║                                              ║"
echo "║   Press Ctrl+C to stop all services         ║"
echo -e "╚══════════════════════════════════════════════╝${NC}"

# Trap Ctrl+C to kill all
trap "echo ''; echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; docker-compose stop; exit 0" INT
wait
