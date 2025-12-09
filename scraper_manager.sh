#!/bin/bash
# ============================================================================
# Bilbasen Scraper Manager for Raspberry Pi 5
# ============================================================================
#
# This script helps manage the bilbasen scraper running in a screen session.
#
# Usage:
#   ./scraper_manager.sh start     - Start the scraper in a screen session
#   ./scraper_manager.sh stop      - Gracefully stop the scraper
#   ./scraper_manager.sh status    - Check if scraper is running
#   ./scraper_manager.sh logs      - View live logs
#   ./scraper_manager.sh progress  - Show current progress
#   ./scraper_manager.sh attach    - Attach to screen session
#
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRAPER_SCRIPT="$SCRIPT_DIR/bilbasen_scraper_pi.py"
OUTPUT_DIR="$SCRIPT_DIR/bilbasen_scrape"
SCREEN_NAME="bilbasen"
LOG_FILE="$OUTPUT_DIR/scraper.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_screen_installed() {
    if ! command -v screen &> /dev/null; then
        print_error "screen is not installed. Install it with: sudo apt install screen"
        exit 1
    fi
}

is_running() {
    screen -list | grep -q "$SCREEN_NAME"
}

start_scraper() {
    check_screen_installed
    
    if is_running; then
        print_warning "Scraper is already running!"
        print_status "Use './scraper_manager.sh attach' to view it"
        exit 1
    fi
    
    # Check if Python script exists
    if [ ! -f "$SCRAPER_SCRIPT" ]; then
        print_error "Scraper script not found: $SCRAPER_SCRIPT"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    print_status "Starting scraper in screen session '$SCREEN_NAME'..."
    
    # Start screen session with the scraper
    screen -dmS "$SCREEN_NAME" bash -c "cd '$SCRIPT_DIR' && python3 '$SCRAPER_SCRIPT' -o '$OUTPUT_DIR' 2>&1; exec bash"
    
    sleep 2
    
    if is_running; then
        print_success "Scraper started successfully!"
        echo ""
        echo "Useful commands:"
        echo "  ./scraper_manager.sh logs      - View live logs"
        echo "  ./scraper_manager.sh progress  - Check progress"
        echo "  ./scraper_manager.sh attach    - Attach to session (Ctrl+A, D to detach)"
        echo "  ./scraper_manager.sh stop      - Stop the scraper"
    else
        print_error "Failed to start scraper"
        exit 1
    fi
}

stop_scraper() {
    if ! is_running; then
        print_warning "Scraper is not running"
        exit 0
    fi
    
    print_status "Sending graceful shutdown signal..."
    
    # Send SIGTERM to the Python process
    screen -S "$SCREEN_NAME" -X stuff $'\003'  # Ctrl+C
    
    # Wait for graceful shutdown
    for i in {1..30}; do
        if ! is_running; then
            print_success "Scraper stopped gracefully"
            exit 0
        fi
        sleep 1
        echo -n "."
    done
    
    echo ""
    print_warning "Scraper didn't stop gracefully, forcing..."
    screen -S "$SCREEN_NAME" -X quit
    
    print_success "Scraper stopped"
}

show_status() {
    echo "=============================================="
    echo "Bilbasen Scraper Status"
    echo "=============================================="
    
    if is_running; then
        print_success "Scraper is RUNNING"
        
        # Show screen info
        echo ""
        echo "Screen session:"
        screen -list | grep "$SCREEN_NAME"
    else
        print_warning "Scraper is NOT running"
    fi
    
    # Show file stats if output exists
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        
        if [ -f "$OUTPUT_DIR/listings.csv" ]; then
            LISTINGS=$(wc -l < "$OUTPUT_DIR/listings.csv")
            echo "Listings collected: $((LISTINGS - 1))"
        fi
        
        if [ -f "$OUTPUT_DIR/car_details.csv" ]; then
            DETAILS=$(wc -l < "$OUTPUT_DIR/car_details.csv")
            echo "Car details collected: $((DETAILS - 1))"
        fi
        
        if [ -d "$OUTPUT_DIR/images" ]; then
            IMAGES=$(ls -1 "$OUTPUT_DIR/images" 2>/dev/null | wc -l)
            echo "Images downloaded: $IMAGES"
        fi
        
        # Disk usage
        echo ""
        echo "Disk usage:"
        du -sh "$OUTPUT_DIR" 2>/dev/null
    fi
}

show_progress() {
    echo "=============================================="
    echo "Bilbasen Scraper Progress"
    echo "=============================================="
    
    # Phase 1 progress
    if [ -f "$OUTPUT_DIR/links_checkpoint.json" ]; then
        echo ""
        echo "Phase 1 (Links):"
        COMPLETED=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/links_checkpoint.json')); print(len(d.get('completed_combos', [])))" 2>/dev/null || echo "?")
        TOTAL=1500  # Approximate total combinations
        echo "  Combinations completed: $COMPLETED / ~$TOTAL"
        
        if [ -f "$OUTPUT_DIR/listings.csv" ]; then
            LISTINGS=$(wc -l < "$OUTPUT_DIR/listings.csv")
            echo "  Listings found: $((LISTINGS - 1))"
        fi
    fi
    
    # Phase 2 progress
    if [ -f "$OUTPUT_DIR/details_checkpoint.json" ]; then
        echo ""
        echo "Phase 2 (Details):"
        PROCESSED=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/details_checkpoint.json')); print(len(d.get('processed_ids', [])))" 2>/dev/null || echo "?")
        
        if [ -f "$OUTPUT_DIR/listings.csv" ]; then
            TOTAL_LISTINGS=$(($(wc -l < "$OUTPUT_DIR/listings.csv") - 1))
            echo "  Cars processed: $PROCESSED / $TOTAL_LISTINGS"
            
            if [ "$PROCESSED" != "?" ] && [ "$TOTAL_LISTINGS" -gt 0 ]; then
                PCT=$((PROCESSED * 100 / TOTAL_LISTINGS))
                echo "  Progress: $PCT%"
            fi
        fi
    fi
    
    # Latest log entries
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "Latest log entries:"
        echo "-------------------"
        tail -5 "$LOG_FILE"
    fi
}

show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        print_error "Log file not found: $LOG_FILE"
        exit 1
    fi
    
    print_status "Showing live logs (Ctrl+C to exit)..."
    echo ""
    tail -f "$LOG_FILE"
}

attach_screen() {
    check_screen_installed
    
    if ! is_running; then
        print_error "Scraper is not running"
        exit 1
    fi
    
    print_status "Attaching to screen session (Ctrl+A, D to detach)..."
    screen -r "$SCREEN_NAME"
}

# ============================================================================
# Main
# ============================================================================

case "$1" in
    start)
        start_scraper
        ;;
    stop)
        stop_scraper
        ;;
    status)
        show_status
        ;;
    progress)
        show_progress
        ;;
    logs)
        show_logs
        ;;
    attach)
        attach_screen
        ;;
    *)
        echo "Bilbasen Scraper Manager"
        echo ""
        echo "Usage: $0 {start|stop|status|progress|logs|attach}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the scraper in a screen session"
        echo "  stop      - Gracefully stop the scraper"
        echo "  status    - Check if scraper is running and show stats"
        echo "  progress  - Show detailed progress information"
        echo "  logs      - View live log output"
        echo "  attach    - Attach to the screen session"
        echo ""
        exit 1
        ;;
esac
