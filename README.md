# Disco

Disco learns disentangled multimodal region representations from heterogeneous urban data for urban region prediction. It organizes modality-specific representations into visual and functional contexts, separates unique and shared information, and identifies cross-region commonality through cosine similarity in the shared feature space. An adaptive Mixture-of-Experts (MoE) module then dynamically fuses the disentangled representations for downstream prediction and interpretable contribution analysis.

Please see the details in our paper below:

- *Disentangled Multimodal Region Representation Learning with Cross-Region Commonality Mining*, accepted by IEEE Transactions on Mobile Computing (TMC).

## Requirements

The code requires the following environment and package versions:

```text
python==3.10.20
numpy==2.2.6
scipy==1.15.3
pandas==2.3.3
scikit-learn==1.7.2
joblib==1.5.3
threadpoolctl==3.6.0
torch==2.12.0
torchvision==0.27.0
networkx==3.4.2
python-louvain==0.16
h3==4.5.0
shapely==2.1.2
Pillow==12.2.0
geopandas==1.1.3
```

### Deom

For training, please run the script as follows:

```bash
python train.py
```

For evaluation, please run the script as follows:

```bash
python evaluate.py
```
## Usage

```bash
raw=raw_data
run=results/my_run
city=my_city

# 1. Map raw data from four modalities to a unified H3 grid
python -m IntraModality.poi.map_h3 "$city" --source-dir "$raw/poi" --regions "$raw/regions.csv" --output-root "$run/processed"
python -m IntraModality.mobility_flow.map_h3 "$city" --source-csv "$raw/flow.csv" --regions "$raw/regions.csv" --output-dir "$run/processed"
python -m IntraModality.map_images_to_h3 streetview "$raw/streetview.csv" "$raw/regions.csv" "$run/processed/$city"
python -m IntraModality.map_images_to_h3 satellite "$raw/satellite.tif" "$raw/regions.csv" "$run/processed/$city"

# 2. Generate embeddings using four modality-specific encoders
python -m IntraModality.poi.main --city "$city" --data-dir "$run/processed" --output-dir "$run/embeddings"
python -m IntraModality.mobility_flow.main --city "$city" --data-dir "$run/processed" --output-dir "$run/embeddings"
python -m IntraModality.streetview.main --city "$city" --data-dir "$run/processed" --output-dir "$run/embeddings"
python -m IntraModality.satellite.main --city "$city" --data-dir "$run/processed" --output-dir "$run/embeddings"

# 3. Align and concatenate multimodal embeddings based on H3 indices
python -m IntraModality.concat --data-dir "$run/embeddings/$city" --output-dir "$run/aligned"

# 4. Disentangle multimodal representations
python -m DisentanglePositive.main --data-dir "$run/aligned" --config config/model.json --output-dir "$run/representations"

# 5. Train and evaluate downstream tasks
python -m DownstreamTask.main --data-dir "$run/representations" --labels "$raw/labels.csv" --label-column target --task regression --config config/downstream.json --output-dir "$run/downstream"
```

## Tips

### 1. Input Data Structure

The raw data directory should follow the structure below:

```text
raw_data/
├── poi/
├── flow.csv
├── streetview.csv
├── satellite.tif
├── regions.csv
└── labels.csv
```

### 2. Required Data Columns

- **`regions.csv`**: Must contain an `h3` column specifying the H3 index of each region.
- **`labels.csv`**: Must contain an `h3` column and a target column for the downstream prediction task.
- **Target column**: Replace `target` in `--label-column target` with the actual target column name in `labels.csv`.
- **H3 consistency**: All modalities and downstream labels should correspond to the same H3 spatial regions to ensure correct alignment.

### 3. Configuration and Parameters

- **`raw`**: Path to the directory containing raw input data.
- **`run`**: Path to the directory where experimental outputs will be saved.
- **`city`**: City identifier used throughout the processing pipeline.
- **`config/model.json`**: Configuration file for representation disentanglement.
- **`config/downstream.json`**: Configuration file for downstream tasks.

### 4. Downstream Tasks

Two task types are supported:

- **Regression**: Use `--task regression`.
- **Classification**: Use `--task classification`.


## Reference

If you use our code, please cite:

```bibtex

```

## Note

