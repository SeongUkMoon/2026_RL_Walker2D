@echo off
REM Saved in CP949 (Korean ANSI) encoding on purpose - see setup.bat.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [실패] .venv 가상환경의 python.exe 를 찾을 수 없습니다.
    echo        먼저 setup.bat 을 실행해서 설치를 끝내 주세요.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\activate.bat" (
    echo [실패] .venv 가상환경의 activate.bat 을 찾을 수 없습니다.
    echo        .venv 폴더를 .venv_old 로 이름을 바꾼 뒤 setup.bat 을 다시 실행하세요.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [실패] .venv 안의 Python을 실행할 수 없거나 Python 3.11이 아닙니다.
    echo        원본 Python이 삭제되거나 이동되어 가상환경이 깨졌을 수 있습니다.
    echo        .venv 폴더를 .venv_old 로 이름을 바꾼 뒤 setup.bat 을 다시 실행하세요.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [실패] .venv 가상환경을 활성화하지 못했습니다. setup.bat 을 다시 실행하세요.
    pause
    exit /b 1
)
python -c "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix and sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [실패] 가상환경 활성화 후의 python 명령이 올바른 .venv를 가리키지 않습니다.
    echo        setup.bat 을 다시 실행하고, 계속되면 강사에게 이 메시지를 보여 주세요.
    pause
    exit /b 1
)
echo ============================================================
echo  Walker2D 실습 터미널  - 가상환경 .venv 활성화됨
echo ------------------------------------------------------------
echo   python 0_check.py                 설치 점검
echo   python 1_run_walker.py            기본 Walker2D 움직여 보기
echo   python 2_build_character.py       내 캐릭터 만들기 - characters/my_character.json
echo   python 3_play.py my_character     내 캐릭터 움직여 보기
echo   python 4_train.py --character my_character --minutes 10     학습
echo   python 5_watch.py --character my_character                  학습 결과 보기
echo   python 6_plot.py --character my_character                   학습 곡선
echo   각 명령 뒤에 --help 를 붙이면 옵션 설명이 나옵니다
echo ============================================================
cmd /k
endlocal
exit /b 0
