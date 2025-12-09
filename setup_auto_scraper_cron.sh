#!/bin/bash
# Setup cron job for auto-scraper on Raspberry Pi
# This script configures the auto-scraper to run every 2 days

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Python executable (adjust if using virtual environment)
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Log file
LOG_FILE="$PROJECT_DIR/auto_scraper_cron.log"

echo "Setting up auto-scraper cron job..."
echo "Project directory: $PROJECT_DIR"
echo "Python binary: $PYTHON_BIN"
echo "Log file: $LOG_FILE"

# Create cron entry (runs at 2 AM every 2 days)
CRON_COMMAND="0 2 */2 * * cd $PROJECT_DIR && $PYTHON_BIN auto_scraper.py --mode incremental >> $LOG_FILE 2>&1"

# Check if cron job already exists
crontab -l 2>/dev/null | grep -q "auto_scraper.py"

if [ $? -eq 0 ]; then
    echo "⚠️  Cron job already exists. Updating..."
    # Remove old entry
    crontab -l 2>/dev/null | grep -v "auto_scraper.py" | crontab -
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -

echo "✅ Cron job installed successfully!"
echo ""
echo "Cron job details:"
echo "  Schedule: Every 2 days at 2:00 AM"
echo "  Command: $CRON_COMMAND"
echo ""
echo "To view cron jobs: crontab -l"
echo "To edit cron jobs: crontab -e"
echo "To remove cron job: crontab -l | grep -v auto_scraper.py | crontab -"
echo ""
echo "Manual test run:"
echo "  cd $PROJECT_DIR && $PYTHON_BIN auto_scraper.py --mode incremental"
