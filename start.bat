@echo off
echo ============================================
echo   GraphRAG Research Paper Assistant
echo   Startup Script (Windows)
echo ============================================
echo.

REM Step 1: Start Docker services
echo [1/4] Starting database services (PostgreSQL, Qdrant, Neo4j, GROBID)...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Docker failed. Is Docker Desktop running?
    pause & exit /b 1
)
echo Waiting for services to be ready (GROBID takes ~60s on first pull)...
timeout /t 15 /nobreak >nul

REM Step 2: Backend
echo [2/4] Setting up backend...
cd backend
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt -q
if not exist uploads mkdir uploads

echo [3/4] Starting backend on :8000...
start "Backend" cmd /k "venv\Scripts\activate && uvicorn app.main:app --reload --port 8000"
cd ..

timeout /t 3 /nobreak >nul

REM Step 3: Frontend
echo [4/4] Starting frontend on :3000...
cd frontend
if not exist node_modules (
    echo Installing npm dependencies...
    npm install
)
start "Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ============================================
echo   Services starting up!
echo.
echo   Frontend:  http://localhost:3000
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo   Neo4j:     http://localhost:7474
echo   GROBID:    http://localhost:8070
echo ============================================
echo.
echo Don't forget to set GEMINI_API_KEY in backend\.env
pause