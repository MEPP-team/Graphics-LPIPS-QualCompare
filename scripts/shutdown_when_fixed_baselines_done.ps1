param(
    [int]$DelaySeconds = 60
)

$ErrorActionPreference = "Stop"
$pattern = "revalidate_fixed_baselines_qualcompare.bat"

Write-Host "[INFO] Watching for running process containing: $pattern"
Write-Host "[INFO] Shutdown delay after completion: $DelaySeconds seconds"
Write-Host "[INFO] To cancel a scheduled shutdown later, run: shutdown /a"

function Get-RevalidationProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*$pattern*" -and
            $_.CommandLine -notlike "*shutdown_when_fixed_baselines_done.ps1*"
        }
}

$seen = $false
while ($true) {
    $processes = @(Get-RevalidationProcesses)
    if ($processes.Count -gt 0) {
        $seen = $true
        $ids = ($processes | ForEach-Object { $_.ProcessId }) -join ", "
        Write-Host "[INFO] Revalidation still running. PIDs: $ids"
        Start-Sleep -Seconds 60
        continue
    }

    if ($seen) {
        Write-Host "[INFO] Revalidation process ended. Scheduling shutdown."
        shutdown /s /t $DelaySeconds /c "Graphics-LPIPS fixed baseline revalidation finished."
        exit 0
    }

    Write-Host "[WARN] No running revalidation process found yet. Checking again in 60 seconds."
    Start-Sleep -Seconds 60
}
