$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildPython = Join-Path $repoRoot ".venv-build\Scripts\python.exe"
$exePath = Join-Path $repoRoot "dist\demo5type.exe"
$hashPath = Join-Path $repoRoot "dist\demo5type.exe.sha256"

Push-Location $repoRoot
try {
    if (-not (Test-Path $buildPython)) {
        python -m venv .venv-build
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the isolated build environment."
        }
    }

    & $buildPython -m pip install --disable-pip-version-check "pyinstaller>=6,<7"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install PyInstaller."
    }

    & $buildPython -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --console `
        --name demo5type `
        --paths $repoRoot `
        scripts/demo5type_cli.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed."
    }

    & $exePath
    if ($LASTEXITCODE -ne 0) {
        throw "The generated executable failed its smoke test."
    }

    $sha256 = (Get-FileHash -Algorithm SHA256 $exePath).Hash.ToLowerInvariant()
    Set-Content -Path $hashPath -Value "$sha256  demo5type.exe" -Encoding ascii
    Write-Host "Built: $exePath"
    Write-Host "SHA-256: $sha256"
}
finally {
    Pop-Location
}
