# Auto-Scraper Cron Setup for Windows Task Scheduler
# This PowerShell script sets up a scheduled task to run auto_scraper.py every 2 days

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$PythonExe = "python"  # Adjust if using virtual environment
$ScriptPath = Join-Path $ProjectDir "auto_scraper.py"
$LogFile = Join-Path $ProjectDir "auto_scraper_cron.log"

Write-Host "Setting up auto-scraper scheduled task..."
Write-Host "Project directory: $ProjectDir"
Write-Host "Python executable: $PythonExe"
Write-Host "Script path: $ScriptPath"
Write-Host "Log file: $LogFile"

# Create scheduled task action
$action = New-ScheduledTaskAction -Execute $PythonExe -Argument "$ScriptPath --mode incremental" -WorkingDirectory $ProjectDir

# Create trigger (every 2 days at 2:00 AM)
$trigger = New-ScheduledTaskTrigger -Daily -DaysInterval 2 -At 2am

# Create scheduled task settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Task name
$taskName = "BPR_AutoScraper"

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "⚠️  Scheduled task already exists. Updating..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Register the scheduled task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Runs Bilbasen auto-scraper every 2 days"

Write-Host "✅ Scheduled task created successfully!"
Write-Host ""
Write-Host "Task details:"
Write-Host "  Name: $taskName"
Write-Host "  Schedule: Every 2 days at 2:00 AM"
Write-Host "  Command: $PythonExe $ScriptPath --mode incremental"
Write-Host ""
Write-Host "To view task: Get-ScheduledTask -TaskName $taskName"
Write-Host "To run task now: Start-ScheduledTask -TaskName $taskName"
Write-Host "To remove task: Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false"
Write-Host ""
Write-Host "Manual test run:"
Write-Host "  cd '$ProjectDir' ; python auto_scraper.py --mode incremental"
