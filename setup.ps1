<#
.SYNOPSIS
    Create the virtual environment, install pinned dependencies, verify the stack.

.DESCRIPTION
    Run from the project root in PowerShell:

        cd C:\Users\uni_2\Documents\Claude\trading-bot
        powershell -ExecutionPolicy Bypass -File .\setup.ps1

    Steps:
      1. Find a Python between 3.11 and 3.14 (the range vectorbt 1.1.0 supports).
      2. Create .venv
      3. Install requirements.txt
      4. Write requirements.lock.txt with the exact resolved versions
      5. Run verify_env.py, which fetches real public data and runs a backtest

    Optional extras (Polymarket public data, JupyterLab):
        .\setup.ps1 -WithOptional

    No API keys or credentials are requested, stored, or configured by this
    script or anything it installs.

    NOTE: this file is deliberately pure ASCII. Windows PowerShell 5.1 reads
    .ps1 files as ANSI unless they carry a UTF-8 BOM, and a mis-decoded dash or
    smart quote turns into a character the parser treats as a string delimiter.
    Please keep it ASCII-only.
#>

param(
    [switch]$WithOptional,
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step {
    param([int]$Number, [string]$Text)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ("{0}. {1}" -f $Number, $Text) -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Get-PythonVersion {
    param([string]$Exe, [string[]]$PreArgs = @())

    $previous = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        $probe = "import sys; print(sys.version_info[0], sys.version_info[1])"
        $output = & $Exe @PreArgs -c $probe 2>&1
        if ($LASTEXITCODE -ne 0) { return $null }

        $text = ($output | Out-String).Trim()
        $parts = $text -split "\s+"
        if ($parts.Count -lt 2) { return $null }

        return [pscustomobject]@{
            Major = [int]$parts[0]
            Minor = [int]$parts[1]
        }
    }
    catch {
        return $null
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Test-Supported {
    param($Version)
    if ($null -eq $Version) { return $false }
    return ($Version.Major -eq 3 -and $Version.Minor -ge 11 -and $Version.Minor -lt 15)
}

# --------------------------------------------------------------- 1. find python
Write-Step 1 "LOCATING A SUPPORTED PYTHON (3.11 to 3.14)"

$PythonCmd = $null
$PythonArgs = @()

# The py launcher lets us ask for an exact version. Preferred order puts the
# versions with the best compiled-wheel coverage first.
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.13", "3.12", "3.11", "3.14")) {
        $found = Get-PythonVersion -Exe "py" -PreArgs @("-$v")
        if (Test-Supported $found) {
            $PythonCmd = "py"
            $PythonArgs = @("-$v")
            Write-Host ("  Using: py -{0}  ->  Python {1}.{2}" -f $v, $found.Major, $found.Minor) -ForegroundColor Green
            break
        }
    }
}

if (-not $PythonCmd) {
    $found = Get-PythonVersion -Exe "python"
    if (Test-Supported $found) {
        $PythonCmd = "python"
        $PythonArgs = @()
        Write-Host ("  Using: python  ->  Python {0}.{1}" -f $found.Major, $found.Minor) -ForegroundColor Green
    }
    elseif ($null -ne $found) {
        Write-Host ("  Found Python {0}.{1} on PATH, which is outside the supported range." -f $found.Major, $found.Minor) -ForegroundColor Yellow
    }
}

if (-not $PythonCmd) {
    Write-Host ""
    Write-Host "No suitable Python found." -ForegroundColor Red
    Write-Host "vectorbt 1.1.0 requires Python 3.11 or newer, and older than 3.15."
    Write-Host "Install Python 3.13 from https://www.python.org/downloads/ then re-run this script."
    Write-Host "During install, tick 'Add python.exe to PATH'."
    exit 1
}

# ------------------------------------------------------------------ 2. venv
Write-Step 2 "CREATING VIRTUAL ENVIRONMENT (.venv)"

if (Test-Path ".venv") {
    Write-Host "  .venv already exists, reusing it."
    Write-Host "  (delete the .venv folder and re-run for a clean rebuild)"
}
else {
    & $PythonCmd @PythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "venv creation failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Created .venv" -ForegroundColor Green
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Expected $VenvPython to exist, but it does not." -ForegroundColor Red
    exit 1
}

# -------------------------------------------------------------- 3. install
Write-Step 3 "INSTALLING DEPENDENCIES"

& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip upgrade failed." -ForegroundColor Red
    exit 1
}

& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Install failed. Most common causes:" -ForegroundColor Red
    Write-Host "  - Python version outside the range vectorbt supports (see step 1)"
    Write-Host "  - numba has no wheel for your Python yet; try Python 3.12 or 3.13"
    Write-Host "  - a proxy or firewall blocking PyPI"
    exit 1
}

if ($WithOptional) {
    Write-Host ""
    Write-Host "  Installing optional extras..."
    & $VenvPython -m pip install -r requirements-optional.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Optional extras failed to install. The core stack is unaffected." -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------------ 4. lock
Write-Step 4 "WRITING requirements.lock.txt"

& $VenvPython -m pip freeze | Out-File -Encoding ascii "requirements.lock.txt"

if (Test-Path "requirements.lock.txt") {
    $lockLines = @(Get-Content "requirements.lock.txt")
    Write-Host ("  {0} exact versions recorded." -f $lockLines.Count) -ForegroundColor Green
    Write-Host "  Commit this file next to any result you want to reproduce later."

    Write-Host ""
    Write-Host "  Key versions:"
    $keys = $lockLines | Select-String -Pattern "^(vectorbt|ccxt|pandas|numpy|pyarrow|matplotlib|numba)=="
    foreach ($line in $keys) {
        Write-Host ("    {0}" -f $line)
    }
}

# ---------------------------------------------------------------- 5. verify
if ($SkipVerify) {
    Write-Host ""
    Write-Host "Skipping verification (-SkipVerify)." -ForegroundColor Yellow
}
else {
    Write-Step 5 "VERIFYING (imports, live public data fetch, real backtest)"
    & $VenvPython verify_env.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Verification failed. Read the output above before continuing." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "SETUP COMPLETE" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host ""
Write-Host "Activate the environment:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run the end-to-end example:"
Write-Host "    python scripts\run_example.py"
Write-Host ""
Write-Host "This environment does research and backtesting. It does not place trades."
