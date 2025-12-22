@ECHO OFF
REM Build documentation script for manuscript-ocr
REM Cleans old build and rebuilds HTML documentation

SETLOCAL

SET SCRIPT_DIR=%~dp0
SET SPHINX_DIR=%SCRIPT_DIR%sphinx
SET BUILD_DIR=%SPHINX_DIR%\_build

ECHO ====================================
ECHO Building manuscript-ocr Documentation
ECHO ====================================
ECHO.

REM Check if virtual environment exists
IF NOT EXIST "%SCRIPT_DIR%..\env\Scripts\sphinx-build.exe" (
    ECHO ERROR: Virtual environment not found or sphinx not installed
    ECHO Please run: pip install sphinx sphinx-rtd-theme numpydoc sphinx-autodoc-typehints sphinxcontrib-mermaid
    EXIT /B 1
)

REM Clean old build
IF EXIST "%BUILD_DIR%" (
    ECHO Cleaning old build directory...
    RMDIR /S /Q "%BUILD_DIR%"
    ECHO Done.
    ECHO.
)

REM Build documentation
ECHO Building HTML documentation...
CD /D "%SPHINX_DIR%"
"%SCRIPT_DIR%..\env\Scripts\sphinx-build.exe" -b html . _build/html

IF %ERRORLEVEL% EQU 0 (
    ECHO.
    ECHO ====================================
    ECHO Build completed successfully!
    ECHO ====================================
    ECHO.
    ECHO Documentation is available at:
    ECHO %BUILD_DIR%\html\index.html
    ECHO.
    ECHO Opening in browser...
    start "" "%BUILD_DIR%\html\index.html"
) ELSE (
    ECHO.
    ECHO ====================================
    ECHO Build failed with errors!
    ECHO ====================================
    EXIT /B 1
)

ENDLOCAL
