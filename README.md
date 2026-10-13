# ModMatchMusic
AI Personalized Playlist Maker

> Generate playlists from free-text mood and genre descriptions, using a model fine-tuned on song lyrics.


## Approach

(Architecture, data, fine-tuning method.)

## Results

(Baseline vs. fine-tuned comparison table, charts, example outputs.)

## Limitations

(Small song catalog, mood labels may be noisy, etc.)

## Getting Started

1. Create and activate a virtual environment:

```bash
   # macOS / Linux / Git Bash
   python3.11 -m venv .venv
   source .venv/bin/activate

   # Windows (cmd)
   py -3.11 -m venv .venv
   .venv\Scripts\activate
```

2. Install dependencies:

```bash
   pip install -r requirements.txt
```

3. Run the tests:

```bash
   pytest
```

## Project Structure

```
├── app/          # Web UI
├── configs/      # Configuration files
├── data/         # Local data (not committed)
├── notebooks/    # Experiments and Colab notebooks
├── results/      # Metrics, tables, and small charts
├── src/          # Source code (data, models, api, utils)
└── tests/        # Unit tests
```

## Data License