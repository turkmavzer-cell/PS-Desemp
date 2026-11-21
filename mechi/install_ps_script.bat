@echo off
setlocal
set "SRC=%~dp0..\tools\photoshop\mechi_studio.jsx"
if not exist "%SRC%" (
    echo HATA: mechi_studio.jsx bulunamadi: "%SRC%"
    pause
    exit /b 1
)

for %%D in (
    "C:\Program Files\Adobe\Adobe Photoshop 2026\Presets\Scripts"
    "C:\Program Files\Adobe\Adobe Photoshop 2021\Presets\Scripts"
) do (
    if exist "%%~D" (
        copy /Y "%SRC%" "%%~D\mechi_studio.jsx" >nul 2>&1
        if errorlevel 1 (
            echo [YETKI GEREKLI] Yonetici olarak calistirin: "%%~D"
        ) else (
            echo [KURULDU] "%%~D\mechi_studio.jsx"
        )
    )
)
echo.
echo Photoshop'ta File ^> Scripts ^> mechi_studio yolundan baslatabilirsiniz.
pause
