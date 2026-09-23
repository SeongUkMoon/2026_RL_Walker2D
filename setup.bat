@echo off
REM ============================================================
REM  Walker2D setup for Korean Windows.
REM  This file is saved in CP949 (Korean ANSI) encoding on purpose:
REM  cmd.exe cannot parse UTF-8 batch files reliably.
REM  If the Korean text looks garbled on your console, the commands
REM  still work - only the messages are affected.
REM ============================================================
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Walker2D 실습 환경 설치  - Python 3.11 + .venv 가상환경
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
    echo [실패] Python 3.11 을 찾을 수 없습니다.   [ERROR] Python 3.11 not found.
    echo.
    echo   1. 아래 주소에서 "Windows installer 64-bit" 를 내려받아 설치하세요.
    echo      https://www.python.org/downloads/release/python-3119/
    echo   2. 설치 첫 화면에서 "Add python.exe to PATH" 를 반드시 체크하세요.
    echo   3. 다른 버전의 Python 이 있어도 3.11 을 추가로 설치해야 합니다.
    echo   4. 설치가 끝나면 이 setup.bat 을 다시 실행하세요.
    echo.
    pause
    exit /b 1
)
echo [1/4] Python 3.11 확인: %PY%

REM ---------- 2) 가상환경 ----------
if exist ".venv\Scripts\python.exe" (
    echo [2/4] 기존 가상환경 .venv 를 확인합니다.
) else (
    echo [2/4] 가상환경 .venv 만드는 중...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [실패] 가상환경을 만들 수 없습니다. 폴더 경로에 한글이나 공백이 없는지 확인하고 다시 실행하세요.
        pause
        exit /b 1
    )
)

REM python.exe 파일이 남아 있어도 원본 Python 삭제/이동 후에는 실행되지 않을 수 있음
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [실패] .venv 안의 Python을 실행할 수 없거나 Python 3.11이 아닙니다.
    echo        깨진 가상환경은 자동으로 삭제하지 않았습니다.
    echo        이 창을 닫고 .venv 폴더를 .venv_old 로 이름을 바꾼 뒤 setup.bat 을 다시 실행하세요.
    echo        문제가 해결된 것을 확인한 뒤 .venv_old 폴더는 직접 삭제해도 됩니다.
    pause
    exit /b 1
)

REM ---------- 3) 패키지 설치 ----------
echo [3/4] 패키지 설치 중: mujoco, gymnasium, stable-baselines3, torch ...   시간이 걸립니다
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo.
    echo [실패] 가상환경의 pip를 실행하거나 업데이트할 수 없습니다.
    echo        위의 .venv 복구 안내에 따라 가상환경을 다시 만든 뒤 재시도하세요.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [실패] 패키지 설치 중 오류가 났습니다. 인터넷 연결을 확인한 뒤 setup.bat 을 다시 실행하세요.
    echo        회사 네트워크 - 프록시나 방화벽 - 라면 개인 핫스팟으로 다시 시도해 보세요.
    pause
    exit /b 1
)

REM ---------- 4) 점검 ----------
echo.
echo [4/4] 설치 점검 실행 - 잠시 후 3초 동안 시뮬레이션 창이 떴다가 닫힙니다
".venv\Scripts\python.exe" 0_check.py
if errorlevel 1 (
    echo.
    echo [실패] 설치 점검을 통과하지 못했습니다. 위의 실패 항목과 힌트를 확인하세요.
    echo        실패 내용을 해결한 다음 setup.bat 또는 python 0_check.py 를 다시 실행하세요.
    pause
    exit /b 1
)
echo.
echo 모든 점검을 통과했습니다. 수업 준비가 끝났습니다.
echo 실습을 시작할 때는 open_terminal.bat 을 더블클릭하세요.
pause
endlocal
exit /b 0
