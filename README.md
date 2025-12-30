# Roof-Ranker

Solar potential analysis for institutional buildings using satellite imagery and AI.

## Overview

Roof-Ranker analyzes satellite imagery of institutional buildings to calculate solar energy potential. It uses Meta's Segment Anything Model (SAM) for roof detection and provides financial and environmental impact estimates.

## Features

- AI roof segmentation using SAM
- Satellite imagery via Google Maps API
- Polygon-based roof selection
- Solar potential calculations
- Financial analysis (savings, ROI)
- CO₂ reduction estimates
- City-specific data for Central Asian capitals

## Installation

### Prerequisites

- Python 3.10+
- Google Maps Static API key

### Using uv

```bash
git clone https://github.com/dudosya/roof-ranker.git
cd roof-ranker
uv sync
uv run streamlit run src/roof_ranker/app.py
```

### Using pip

```bash
pip install -r requirements.txt
streamlit run src/roof_ranker/app.py
```

Note: First run downloads SAM model weights (~350MB).

## Usage

1. Run: `streamlit run src/roof_ranker/app.py`
2. Enter Google Maps API key
3. Enter building address
4. Select city for location data
5. Click "Fetch Satellite Image"
6. Draw polygon around roof area
7. Click "Confirm Segmentation"
8. View analysis results

## Project Structure

```
roof-ranker/
├── src/roof_ranker/
│   ├── app.py                 # Main application
│   ├── utils/
│   │   ├── constants.py       # City data
│   │   ├── vision.py          # Image processing
│   │   ├── neural_vision.py   # SAM integration
│   │   ├── calculator.py      # Solar calculations
│   │   └── geo_handler.py     # Maps API
│   └── assets/               # Sample images
├── tests/                    # Unit tests
├── pyproject.toml           # Project config
└── README.md
```

## Development

### Tests

```bash
uv run pytest tests/
```

### Code Quality

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/
```

## Tech Stack

- Python 3.10+
- Streamlit
- PyTorch + SAM
- OpenCV
- Google Maps API

## License

MIT
