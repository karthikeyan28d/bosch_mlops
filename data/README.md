# Data Directory Setup

This directory contains the multimodal biometric dataset.

## Automated Download (Recommended)

The dataset can be automatically downloaded from Kaggle:

```bash
# Download using run.py
python run.py --mode download

# Or use the data_download module directly
python -m src.data_download --output data/raw

# Force re-download
python run.py --mode download --force
```

### Kaggle Setup
Before downloading, configure your Kaggle credentials:

**Option A: Create kaggle.json file**
1. Go to https://www.kaggle.com/settings
2. Click 'Create New Token'
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`

**Option B: Environment variables**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

## Manual Download

Alternatively, download manually:
1. Download from: https://www.kaggle.com/datasets/ninadmehendale/multimodal-iris-fingerprint-biometric-data
2. Extract the contents to `data/raw/`

## Expected Structure

After download/extraction, the structure should look like:

```
data/
└── raw/
    ├── iris/
    │   ├── 001_L_01.bmp
    │   ├── 001_L_02.bmp
    │   ├── 001_R_01.bmp
    │   └── ... (more iris images)
    └── fingerprint/
        ├── 001_L_thumb_01.BMP
        ├── 001_L_index_01.BMP
        └── ... (more fingerprint images)
```

## Filename Convention

The dataset uses the following naming convention:
- `{subject_id}_{eye/hand}_{sample_number}.bmp`
- Subject IDs are 3-digit numbers (001-040)
- Eye: L (left), R (right)
- Hand: L (left), R (right)

## Data Processing

Once data is in place, run preprocessing:
```bash
python run.py --mode preprocess
```

Or run the complete pipeline:
```bash
python run.py --mode all
```
