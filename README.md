# Nepal Earthquake Damage Prediction

Predicts building damage grade (1=low, 2=medium, 3=severe) after the 2015 Nepal earthquake, based on structural and location features.

## Dataset
[Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/) (DrivenData). Download `train_values.csv` and `train_labels.csv` and place them in `data/raw/`.

## Project structure
- `src/data_loader.py` — loads and merges raw data
- `src/eda.py` — exploratory data analysis
- `src/preprocess.py` — cleaning, encoding, train/test split
- `src/train.py` — trains and compares Logistic Regression, Decision Tree, Random Forest

## Results

| Model | F1 Macro | F1 Weighted | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.597 | 0.665 | 0.68 |
| Decision Tree | 0.605 | 0.672 | 0.68 |
| **Random Forest (best)** | **0.608** | **0.672** | **0.69** |

## Usage
\`\`\`bash
python -m src.preprocess
python -m src.train
\`\`\`

## Model & Demo
- Trained model: [HF Model Repo link - TBD]
- Live demo: [HF Space link - TBD]