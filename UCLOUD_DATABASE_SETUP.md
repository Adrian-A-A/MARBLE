# Database Scenario Setup for UCloud

## Overview
The MARBLE database scenario requires container services (PostgreSQL, Prometheus, exporters) to run. This guide explains how to set this up on UCloud, which doesn't support native Docker.

## Solution: udocker

UCloud supports `udocker` - a user-space Docker implementation that doesn't require root privileges or kernel-level container support.

### Installation

**Option 1: Automatic (Recommended)**
```bash
pip install udocker
```

**Option 2: Manual**
Download from: https://github.com/indigo-dc/udocker

### Verification

```bash
udocker --version
udocker setup
```

## How It Works

The modified `marble/environments/db_env.py` now:

1. **Detects available runtime** at startup
   - Tries Docker with sudo first (for systems that have it)
   - Falls back to `udocker` if Docker isn't available
   - Gracefully continues even if neither is available

2. **Automatically selects the appropriate tool**
   - No code changes needed when switching between environments

3. **Handles failures gracefully**
   - Container services (Prometheus, exporters) are optional
   - Direct database access is the critical component
   - If containers fail to start, the pipeline continues with a warning

## Running the Database Scenario

### On UCloud with udocker:
```bash
# Prerequisites
pip install udocker

# Run the scenario
python scripts/pipeline/run_vllm_pipeline.py \
  --scenarios database \
  --models openai/Qwen/Qwen3.5-4B
```

### On systems with Docker:
```bash
# No setup needed - Docker is detected automatically
python scripts/pipeline/run_vllm_pipeline.py \
  --scenarios database \
  --models openai/Qwen/Qwen3.5-4B
```

## What Changed

### `marble/environments/db_env.py`
- Added `_detect_container_runtime()` method to detect Docker or udocker
- Modified `start_docker_containers()` to support both runtimes
- Modified `terminate()` to support both runtimes
- Added graceful error handling if neither runtime is available
- Added `shutil` import for executable detection

### `scripts/pipeline/vllm_pipeline.yaml`
- Removed hard requirement for `docker` command
- Added explanatory comment about UCloud/udocker usage

## Limitations and Workarounds

### udocker Considerations
- `udocker` is slower than native Docker
- Networking between containers works but may be less efficient
- Some advanced Docker features may not be available

### If Container Services Fail
- PostgreSQL database connection is the critical requirement
- Prometheus monitoring (alerts, metrics) will be unavailable
- The scenario can still complete with SQL query analysis only
- Retry with more verbose logging: `--debug` or check logs

## Troubleshooting

### "Cannot find docker or udocker"
```bash
# Install udocker
pip install udocker

# Verify installation
which udocker
udocker --version
```

### "udocker setup fails"
```bash
# Run setup with explicit tarball
udocker install
```

### "PostgreSQL connection refused"
```bash
# Check if containers are running
udocker ps

# Check logs
udocker logs <container_name>

# Manually troubleshoot
udocker run -it postgres:latest psql -U test -h localhost
```

## Files Modified
- `marble/environments/db_env.py` - Container runtime detection and fallback
- `scripts/pipeline/vllm_pipeline.yaml` - Removed Docker requirement

## Next Steps
1. Install udocker: `pip install udocker`
2. Run database scenario: `python scripts/pipeline/run_vllm_pipeline.py --scenarios database --models openai/Qwen/Qwen3.5-4B`
3. Monitor output for any container issues and adjust as needed
