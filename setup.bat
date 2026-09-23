@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Walker2D 실습 환경 설치  (Python 3.11 + .venv 가상환경)
echo  인터넷이 필요하며 5~10분 정도 걸립니다. 창을 닫지 말고 기다려 주세요.
echo ============================================================
echo.

REM ---------- 1) Python 3.11 찾기 ----------
set "PY="
py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [실패] Python 3.11 을 찾을 수 없습니다.
    echo.
    echo   1. https://www.python.org/downloads/release/python-3119/ 에서
    echo      "Windows installer (64-bit)" 를 내려받아 설치하세요.
    echo   2. 설치 첫 화면에서 "Add python.exe to PATH" 를 반드시 체크하세요.
    echo   3. 설치가 끝나면 이 setup.bat 을 다시 실행하세요.
    echo.
    pause
    exit /b 1
)
echo [1/4] Python 3.11 확인: %PY%

REM ---------- 2) 가상환경 ----------
if exist ".venv\Scripts\python.exe" (
    echo [2/4] 가상환경 .venv 가 이미 있습니다. 그대로 사용합니다.
) else (
    echo [2/4] 가상환경 .venv 만드는 중...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [실패] 가상환경을 만들 수 없습니다. 폴더 경로에 한글이 없는지 확인하고 다시 실행하세요.
        pause
        exit /b 1
    )
)

REM ---------- 3) 패키지 설치 ----------
echo [3/4] 패키지 설치 중 (mujoco, gymnasium, stable-baselines3, torch ...)  -- 시간이 걸립니다
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [실패] 패키지 설치 중 오류가 났습니다. 인터넷 연결을 확인한 뒤 setup.bat 을 다시 실행하세요.
    echo        회사 네트워크(프록시/방화벽)라면 개인 핫스팟으로 다시 시도해 보세요.
    pause
    exit /b 1
)

REM ---------- 4) 점검 ----------
echo.
echo [4/4] 설치 점검 실행 (잠시 후 3초 동안 시뮬레이션 창이 떴다가 닫힙니다)
".venv\Scripts\python.exe" 0_check.py
echo.
echo 위에 "모든 점검 통과" 가 보이면 준비 완료입니다.
echo 실습을 시작할 때는 open_terminal.bat 을 더블클릭하세요.
pause
