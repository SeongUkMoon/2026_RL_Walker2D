@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo 먼저 setup.bat 을 실행해서 설치를 끝내 주세요.
    pause
    exit /b 1
)
echo ============================================================
echo  Walker2D 실습 터미널  (가상환경 .venv 활성화됨)
echo ------------------------------------------------------------
echo   python 0_check.py                 설치 점검
echo   python 1_run_walker.py            기본 Walker2D 움직여 보기
echo   python 2_build_character.py       내 캐릭터 만들기 (characters/my_character.json)
echo   python 3_play.py my_character     내 캐릭터 움직여 보기
echo   python 4_train.py --character my_character --minutes 10     학습
echo   python 5_watch.py --character my_character                  학습 결과 보기
echo   python 6_plot.py --character my_character                   학습 곡선
echo   (각 명령 뒤에 --help 를 붙이면 옵션 설명이 나옵니다)
echo ============================================================
cmd /k ".venv\Scripts\activate.bat"
