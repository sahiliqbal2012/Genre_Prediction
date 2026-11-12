# MM-IMDb Multimodal Movie Genre Classification

## Project Overview

This project investigates multimodal machine learning for movie genre classification using the MM-IMDb dataset. The objective is to predict movie genres by combining information from movie posters and plot summaries.

The project compares four approaches:
- Poster-only classification
- Text-only classification
- Early fusion
- Late fusion

The main hypothesis is that combining visual and textual information can improve genre prediction compared with using either modality independently.

## Dataset

The project uses the MM-IMDb dataset, containing 25,959 movies and 23 genre labels.

The implementation uses pre-extracted features:
- Poster features: 4,096 dimensions
- Text features: 300 dimensions

The dataset was divided into:
- Training: 15,552 samples
- Validation: 2,608 samples
- Testing: 7,799 samples

The original dataset is not included in this repository. It can be obtained from the Kaggle MM-IMDb dataset source:

https://www.kaggle.com/datasets/johnarevalo/mmimdb

## Methodology

The implementation uses PyTorch to train feed-forward neural networks for multi-label classification.

### Poster-only model
Uses the 4,096-dimensional poster feature vectors to predict the 23 movie genres.

### Text-only model
Uses the 300-dimensional plot feature vectors to predict the movie genres.

### Early fusion
Combines poster and text features by concatenating them before passing them through a shared neural network.

### Late fusion
Combines the prediction probabilities from the independently trained poster and text models. The fusion weight and classification threshold were selected using validation data.

## Preprocessing

- Checked the dataset for missing values.
- Applied StandardScaler separately to poster and text features.
- Fitted scalers using training data only.
- Applied the fitted transformations to validation and test data.
- Used binary cross-entropy with logits for multi-label classification.

## Evaluation Metrics

The models were evaluated using:
- Micro-F1
- Macro-F1
- Micro Average Precision
- Macro Average Precision
- Hamming Loss

## Results

| Model | Micro-F1 | Macro-F1 |
|---|---:|---:|
| Poster-only | 0.4037 | 0.2352 |
| Text-only | 0.5771 | 0.4459 |
| Early Fusion | 0.5745 | 0.4591 |
| Late Fusion | 0.6255 | 0.5410 |

Late fusion achieved the highest Micro-F1 and Macro-F1 scores among the four approaches. Text-only classification achieved the lowest Hamming Loss.

## Repository Structure

- `notebooks/` – Implementation notebook.
- `models/` – Trained model checkpoints, scalers and late-fusion settings.
- `outputs/` – Model comparison and per-genre evaluation results.
- `figures/` – Performance charts and training loss plots.

## How to Reproduce

1. Download the MM-IMDb dataset from the Kaggle source listed above.
2. Open the implementation notebook in Kaggle or a compatible Python environment.
3. Configure the dataset path to point to the downloaded HDF5 file.
4. Install the required Python packages.
5. Run the notebook cells in sequence to perform preprocessing, model training, validation and testing.
6. Review the generated evaluation metrics, CSV files and visualisations.

The notebook uses a fixed random seed for the experiment.

## Limitations

The project uses pre-extracted features rather than training image and text encoders from raw movie posters and plot summaries. Performance also varies across genres, particularly for less frequent categories. The results should therefore be interpreted in the context of class imbalance and the selected experimental setup.

## Ethical Considerations

The project considers potential representation bias in the dataset, genre imbalance and possible biases in pre-trained feature representations. The system is intended for academic experimentation and should not be treated as a definitive classification of a movie.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- h5py
- Jupyter Notebook / Kaggle
