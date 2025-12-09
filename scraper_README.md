# Bilbasen Scraper for Raspberry Pi 5

A robust web scraper designed to run for several days on a Raspberry Pi 5, collecting ~45,000 car listings from bilbasen.dk.

## Features

- **Two-phase scraping**: First collects all listing URLs, then extracts detailed car information
- **Resume capability**: Can resume from where it left off after crashes or restarts
- **Screen session**: Runs in background, survives SSH disconnections
- **Progress logging**: Detailed logs with ETA estimates
- **Graceful shutdown**: Saves checkpoint on Ctrl+C or system signals
- **Image download**: Downloads car images linked by external_id

## Quick Start

### 1. Transfer files to Raspberry Pi

```bash
# From your computer
scp bilbasen_scraper_pi.py scraper_manager.sh pi@<raspberry-pi-ip>:~/bilbasen/
```

### 2. SSH into Raspberry Pi

```bash
ssh pi@<raspberry-pi-ip>
cd ~/bilbasen
```

### 3. Install dependencies

```bash
# Install screen if not already installed
sudo apt update
sudo apt install screen python3-pip

# Install Python packages
pip3 install requests pandas
```

### 4. Make scripts executable

```bash
chmod +x bilbasen_scraper_pi.py
chmod +x scraper_manager.sh
```

### 5. Start the scraper

```bash
./scraper_manager.sh start
```

### 6. Close SSH and let it run

You can now safely close your SSH session. The scraper will continue running.

## Managing the Scraper

```bash
# Check if running and view stats
./scraper_manager.sh status

# View detailed progress
./scraper_manager.sh progress

# View live logs
./scraper_manager.sh logs

# Attach to screen session (see output)
./scraper_manager.sh attach
# (Press Ctrl+A, then D to detach)

# Stop the scraper gracefully
./scraper_manager.sh stop
```

## Manual Screen Commands

```bash
# Start a new screen session
screen -S bilbasen

# Run the scraper
python3 bilbasen_scraper_pi.py

# Detach from screen: Press Ctrl+A, then D

# List screen sessions
screen -ls

# Reattach to session
screen -r bilbasen

# Kill a screen session
screen -S bilbasen -X quit
```

## Command Line Options

```bash
# Run with default settings (both phases)
python3 bilbasen_scraper_pi.py

# Specify output directory
python3 bilbasen_scraper_pi.py -o /path/to/output

# Run only Phase 1 (links scraping)
python3 bilbasen_scraper_pi.py -p 1

# Run only Phase 2 (details scraping)
python3 bilbasen_scraper_pi.py -p 2

# Skip image downloads
python3 bilbasen_scraper_pi.py --no-images

# Test mode (limited scraping)
python3 bilbasen_scraper_pi.py --test
```

## Output Files

```
bilbasen_scrape/
├── scraper.log           # Detailed log file
├── listings.csv          # Phase 1: All listing URLs
├── links_checkpoint.json # Phase 1: Resume checkpoint
├── car_details.csv       # Phase 2: All car details (40+ columns)
├── details_checkpoint.json # Phase 2: Resume checkpoint
└── images/               # Downloaded car images
    ├── 6673145.jpg
    ├── 6590920.jpg
    └── ...
```

## Expected Runtime

- **Phase 1** (Links): ~6-12 hours for ~1,500 filter combinations
- **Phase 2** (Details): ~2-4 days for ~45,000 car pages + images
- **Total**: ~3-5 days depending on network speed and delays

## Monitoring Progress

### From another terminal:

```bash
# Quick status
./scraper_manager.sh status

# Detailed progress
./scraper_manager.sh progress

# Live logs
tail -f bilbasen_scrape/scraper.log

# Count collected items
wc -l bilbasen_scrape/listings.csv
wc -l bilbasen_scrape/car_details.csv
ls bilbasen_scrape/images | wc -l
```

### Check disk space:

```bash
df -h
du -sh bilbasen_scrape/
```

## Resuming After Crash

The scraper automatically resumes from checkpoints. Just restart it:

```bash
./scraper_manager.sh start
```

Or manually:

```bash
screen -S bilbasen
python3 bilbasen_scraper_pi.py
# It will detect existing checkpoints and continue
```

## Troubleshooting

### Scraper stops unexpectedly

1. Check the log file:
   ```bash
   tail -100 bilbasen_scrape/scraper.log
   ```

2. Check for disk space issues:
   ```bash
   df -h
   ```

3. Restart the scraper (it will resume):
   ```bash
   ./scraper_manager.sh start
   ```

### Network errors

The scraper has built-in retry logic (5 retries with exponential backoff). If you see many network errors:

1. Check your internet connection
2. The website might be rate-limiting you - the delays are configured to be respectful

### Memory issues

The scraper is optimized for low memory usage, processing one item at a time. If you still have issues:

```bash
# Check memory usage
free -h

# Restart the Pi and resume
sudo reboot
# After reboot:
./scraper_manager.sh start
```

### Permission denied

```bash
chmod +x bilbasen_scraper_pi.py
chmod +x scraper_manager.sh
```

## CSV Column Reference

### car_details.csv columns (40+):

**Core:**
- `external_id`, `url`, `brand`, `model`, `variant`, `title`, `price`
- `description`, `equipment`

**Details:**
- `details_model_year`, `details_first_registration`, `details_mileage_km`
- `details_fuel_type`, `details_range_km`, `details_battery_capacity_kwh`
- `details_power_hp_nm`, `details_acceleration_0_100`, `details_top_speed`
- `details_color`, `details_fuel_consumption`, `details_co2_emission`
- And more...

**Model Info:**
- `model_new_price`, `model_body_type`, `model_weight_kg`
- `model_length_cm`, `model_width_cm`, `model_height_cm`
- And more...

**Seller:**
- `seller_name`, `seller_type`, `seller_city`, `seller_zipcode`

**Dates:**
- `listing_date`, `last_updated`, `registration_date`

**Images:**
- `image_url`, `image_filename`

## Tips for Long-Running Scrapes

1. **Use a stable power supply** - Consider a UPS for the Raspberry Pi

2. **Use wired ethernet** - More stable than WiFi for multi-day operations

3. **Monitor temperature**:
   ```bash
   vcgencmd measure_temp
   ```

4. **Set up email notifications** (optional):
   Add to the end of bilbasen_scraper_pi.py:
   ```python
   # Send email when done
   import smtplib
   # ... email code
   ```

5. **Backup checkpoints periodically**:
   ```bash
   # Add to crontab
   0 */6 * * * cp -r ~/bilbasen/bilbasen_scrape ~/bilbasen_backup_$(date +\%Y\%m\%d)
   ```
