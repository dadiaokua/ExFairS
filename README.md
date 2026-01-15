# ExFairS

**Experiential Fairness Scheduling for Large Language Model Inference Services**

## Overview

ExFairS introduces **Experiential Fairness** - a user-centric approach to fairness optimization in LLM serving systems. Unlike traditional schedulers that focus solely on system-level metrics, ExFairS formulates a composite metric that unifies user SLO compliance with system resource consumption to guide scheduling decisions.

### Key Innovation

- **User-Centric Fairness**: Bridges the gap between system metrics and actual user experience
- **Real-time Monitoring**: Continuous monitoring with dynamic priority adjustment
- **Composite Metric**: Unifies SLO compliance with resource consumption
- **Proven Performance**: Reduces SLO violation rates while improving fairness

## Quick Start

### Installation

```bash
git clone https://github.com/dadiaokua/ExFairS.git
cd ExFairS
pip install -r requirements.txt
```

### Run Experiments

```bash
# Run single scenario
./run.sh --scenario scenario_I --strategy QUEUE_ExFairS

# Run all strategies for comparison
./run.sh --scenario scenario_I

# Run all scenarios
./run.sh --all
```

### View Results

```bash
# Visualize results
python3 scripts/visualize_results.py scenario_I
```

## Real-time Monitoring System

ExFairS uses a real-time monitoring architecture instead of round-based evaluation:

```
T0 ─────── T60s ─────── T120s ─────── T180s ─────── ... ─────── T600s
   │         │            │             │                        │
   │ Window1 │  Window2   │   Window3   │   ...                  │ End
   │ Collect │  Collect   │   Collect   │                        │
   │ SAFI    │  SAFI      │   SAFI      │                        │
   │ Adjust  │  Adjust    │   Adjust    │                        │
```

### Key Features

1. **Continuous Monitoring**: Background thread collects data every 60 seconds
2. **Real-time SAFI Calculation**: Fairness metrics updated without blocking inference
3. **Dynamic Alpha Adjustment**: Weight parameter adapts based on fairness variance
4. **Immediate Priority Update**: New priorities applied instantly to incoming requests

### Core Components

- `RealtimeMonitor`: Main monitoring class
- `ClientStats`: Window-based client statistics
- `RequestQueueManager`: Queue management with scheduling strategies

## Scheduling Strategies

| Strategy | Description | Focus |
|----------|-------------|-------|
| **ExFairS** | Experiential Fairness Scheduling | User experience + efficiency |
| **Justitia** | Virtual time-based scheduling | Short job prioritization |
| **SLOGreedy** | SLO violation greedy | SLO compliance |
| **VTC** | Variable Token Credits | Token fairness |
| **FCFS** | First Come First Serve | Simplicity |

## Scenarios

| Scenario | Description | Clients |
|----------|-------------|---------|
| `scenario_I` | Balanced load | 2 Mix clients |
| `scenario_II` | Imbalanced load | 2 Mix clients |
| `scenario_III` | Heterogeneous 4 | 4 Mix clients |
| `scenario_IV` | Heterogeneous 8 | 8 Mix clients |
| `scenario_V` | High concurrency | 20 Mix clients |

## Configuration

### Scenario Configuration (`config/scenarios/*.yaml`)

```yaml
name: scenario_I
clients:
  - type: Mix
    count: 1
    qpm: 20
    slo: 20
  - type: Mix
    count: 1
    qpm: 60
    slo: 20
```

### Global Configuration (`config/Config.py`)

```python
GLOBAL_CONFIG = {
    "alpha": 0.8,                    # Initial alpha weight
    "fairness_ratio_exfairs": 0.05,  # Exchange threshold
    "max_exchange_times": 3,         # Max exchanges per window
    "ADJUST_SENSITIVITY": 2.0,       # Priority adjustment sensitivity
}
```

## Results Structure

```
results/run_<timestamp>/
├── metadata.json
├── run.log
└── <scenario>/
    ├── <strategy>/
    │   ├── results.json             # Final statistics
    │   ├── benchmark_results.json   # Monitor history
    │   ├── realtime_metrics_*.png   # Trend visualization
    │   └── performance_metrics_*.png
    └── charts/
        └── performance.png          # Strategy comparison
```

## Project Structure

```
ExFairS/
├── config/
│   ├── scenarios/           # Scenario configurations
│   └── vllm/               # vLLM engine config
├── BenchmarkClient/        # Client implementation
├── BenchmarkMonitor/       # RealtimeMonitor
├── RequestQueueManager/    # Queue and scheduling
├── scripts/                # Benchmark and visualization
├── util/                   # Utility functions
├── results/                # Experiment results
└── run.sh                  # Main entry point
```

## Documentation

- [Quick Start](docs/QUICKSTART.md) - 5-minute getting started
- [Visualization Guide](docs/Visualization_Guide.md) - Result analysis
- [Ubuntu Setup](docs/UBUNTU_SETUP_GUIDE.md) - Environment setup

## Requirements

- Python 3.7+
- PyTorch
- vLLM
- transformers
- matplotlib
- numpy

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
