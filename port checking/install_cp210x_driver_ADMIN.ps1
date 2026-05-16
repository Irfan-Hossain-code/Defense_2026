# Install Silicon Labs CP210x driver for ESP32 (CP2102N)
# This script self-elevates to Administrator if needed.

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator")) {
    Write-Host "Re-launching as Administrator..."
    Start-Process powershell -Verb RunAs -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

Write-Host "=== CP210x Driver Installer ===" -ForegroundColor Cyan
Write-Host "Running as Administrator."

$tmpDir = Join-Path $env:TEMP "cp210x_driver"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$zipPath  = Join-Path $tmpDir "CP210x_drivers.zip"
$url      = "https://www.silabs.com/documents/public/software/CP210x_Windows_Drivers.zip"

Write-Host "`nDownloading CP210x drivers from Silicon Labs..."
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    Write-Host "Download complete." -ForegroundColor Green
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    Write-Host "`nPlease download manually from: https://www.silabs.com" -ForegroundColor Yellow
    Write-Host "Search for: CP210x VCP drivers"
    pause
    exit 1
}

Write-Host "Extracting..."
Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force

# Find the 64-bit installer
$installer = Get-ChildItem -Path $tmpDir -Recurse -Filter "CP210xVCPInstaller_x64.exe" | Select-Object -First 1
if (-not $installer) {
    $installer = Get-ChildItem -Path $tmpDir -Recurse -Filter "*.exe" | Select-Object -First 1
}

if ($installer) {
    Write-Host "`nRunning installer: $($installer.FullName)"
    Start-Process -FilePath $installer.FullName -Wait
    Write-Host "`nInstaller finished." -ForegroundColor Green
} else {
    # Fall back to pnputil with the INF file
    Write-Host "No EXE found — trying pnputil with INF files..."
    $infFiles = Get-ChildItem -Path $tmpDir -Recurse -Filter "*.inf"
    foreach ($inf in $infFiles) {
        Write-Host "  Installing: $($inf.FullName)"
        pnputil /add-driver $inf.FullName /install
    }
}

Write-Host "`nChecking if ESP32 is now visible..."
$device = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.InstanceId -like "*VID_10C4*" }
if ($device) {
    Write-Host "Device found: $($device.FriendlyName) - Status: $($device.Status)" -ForegroundColor Green
    if ($device.Status -ne "OK") {
        Write-Host "Enabling device..."
        Enable-PnpDevice -InstanceId $device.InstanceId -Confirm:$false
    }
} else {
    Write-Host "Device not visible yet — try unplugging and replugging the ESP32." -ForegroundColor Yellow
}

Write-Host "`nDone! Replug your ESP32 if it is not detected yet." -ForegroundColor Cyan
pause
