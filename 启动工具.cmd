@echo off
chcp 65001 >nul
if exist "%~dp0可直接运行\HSV阈值调试工具\HSV阈值调试工具.exe" (
    start "" "%~dp0可直接运行\HSV阈值调试工具\HSV阈值调试工具.exe"
    exit /b
)
if exist "D:\Anconda\envs\yolo\python.exe" (
    "D:\Anconda\envs\yolo\python.exe" "%~dp0hsv_tool.py"
) else (
    python "%~dp0hsv_tool.py"
)
pause
