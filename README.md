# Janus Bitcoin Price Regressor

A state-of-the-art Bitcoin price change prediction system using the Mamba (Selective State Space Model) architecture. This project predicts continuous future Bitcoin price changes based on historical market data, enabling precise forecasting for trading applications.

## 🎯 Project Overview

This regressor analyzes Bitcoin market data and predicts continuous price changes over the next 20 candles (15-minute timeframe), outputting:

- **Continuous percentage change**: Direct prediction of future price movement
- **Stop-loss aware**: Predictions adjusted for realistic trading scenarios
- **Direction and magnitude**: Both trend direction and expected size of movement

## 🏗️ Architecture

### Mamba SSM (State Space Model)

The project implements the Mamba architecture, which offers:
- **Linear-time complexity** for sequence processing
- **Superior long-range dependency modeling** compared to traditional RNNs
- **Efficient training and inference** on long sequences
- **Selective state space mechanism** for adaptive information flow

### Model Components

1. **Input Projection Layer**: Maps 13 input features to model dimension
2. **Stacked Mamba Blocks**: Multiple layers with residual connections
3. **Layer Normalization**: Stabilizes training
4. **Regression Head**: Final prediction layer outputting continuous price change

## 📊 Dataset

### Features (13 total)

**M15 Timeframe:**
- RSI_14_M15
- ATR_5_pct_M15
- dist_from_ema_10_M15
- ema10_slope_M15
- volume_oscillator_M15
- obv_M15
- hour_sin, hour_cos (temporal encoding)

**Higher Timeframes:**
- RSI_14_H1
- ADX_14_H1
- EMA_diff_21_H4pct
- RSI_15_x_NYHours (interaction feature)
- garch_volatility

### Input Format

- **Sequence Length**: 64 candles (16 hours of 15-min data)
- **Prediction Horizon**: 20 candles ahead (5 hours)
- **Input Shape**: (batch_size, 64, 13)
- **Output Shape**: (batch_size, 1) continuous price change percentage
- **Normalization**: All features scaled to [0, 1]
- **Stop Loss Awareness**: Dynamic ATR-based (1.4x ATR) integrated into labels
- **Target**: Realistic price change accounting for intraday stop-loss hits

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Entry Points

The project has three main entry points:

#### 1. Dataset Creation
Create the Janus dataset from raw BTC CSV data:

```bash
cd dataset
python -m create_dataset
```

**Input:** `btc.csv` (raw OHLCV data)  
**Output:** `janus_m15_dataset.parquet`, `janus_m15_scaler.joblib`

#### 2. Model Training
Train the Mamba regressor on the created dataset:

```bash
python train.py --config config.yaml
```

**Optional arguments:**
- `--data-path`: Override data path from config
- `--output-dir`: Override output directory
- `--resume`: Path to checkpoint to resume training

**For Kaggle:**
```bash
python kaggle_train.py
```

#### 3. Backtesting
Run backtest on trained model:

```bash
python backtest.py \
  --checkpoint checkpoints/best_model.pt \
  --data dataset/janus_m15_dataset.parquet \
  --start-date 2025-08-01 \
  --end-date 2025-09-30 \
  --initial-capital 6000000 \
  --leverage 5
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--data`: Path to dataset parquet file (required)
- `--start-date`: Backtest start date
- `--end-date`: Backtest end date
- `--initial-capital`: Initial capital in USD
- `--leverage`: Leverage multiplier

## 📁 Project Structure

```
V5/
├── config.yaml                 # Model training configuration
├── requirements.txt            # Python dependencies
├── train.py                    # Main training script ⭐
├── backtest.py                 # Backtesting script ⭐
├── kaggle_train.py            # Kaggle-specific training script
├── README.md                   # This file
│
├── src/
│   ├── config/
│   │   └── config_loader.py   # Configuration management
│   │
│   ├── data/
│   │   ├── base_strategy.py   # Strategy pattern interface
│   │   ├── sequence_strategy.py # Sequence processing strategy
│   │   ├── dataset.py         # PyTorch dataset (regression)
│   │   └── data_loader.py     # Data loader factory
│   │
│   ├── models/
│   │   ├── mamba_block.py     # Mamba SSM block implementation
│   │   └── mamba_regressor.py # Complete regressor model
│   │
│   ├── training/
│   │   └── trainer.py         # Training loop with early stopping
│   │
│   ├── evaluation/
│   │   ├── evaluator.py       # Regression evaluation metrics
│   │   └── visualizer.py      # Visualization utilities
│   │
│   └── utils/
│       └── helpers.py         # Utility functions
│
├── dataset/
│   ├── config.py              # Dataset configuration
│   ├── create_dataset.py      # Dataset creation script ⭐
│   ├── data_processor.py      # Multi-timeframe processor
│   ├── indicators.py          # Technical indicator calculator
│   ├── labeling.py            # Regression labeling strategy
│   ├── janus_m15_dataset.parquet
│   └── janus_m15_scaler.joblib
│
└── backtest/
    ├── config.py              # Backtest configuration
    ├── engine.py              # Backtesting engine
    ├── position.py            # Position management
    ├── trade.py               # Trade record
    └── reporter.py            # Rich terminal reporting
```

## ⚙️ Configuration

Key configuration parameters in `config.yaml`:

### Model Parameters
```yaml
model:
  name: "MambaBTC_Regression"
  d_model: 128          # Model dimension
  d_state: 16           # SSM state dimension
  d_conv: 4             # Convolution kernel size
  n_layers: 4           # Number of Mamba blocks
  dropout: 0.3          # Dropout rate
  num_classes: 1        # Output dimension (1 for regression)
```

### Training Parameters
```yaml
training:
  epochs: 100
  learning_rate: 0.0005
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "reduce_on_plateau"
  scheduler_patience: 5
  scheduler_factor: 0.5
  early_stopping_patience: 15
  gradient_clip: 1.0

loss:
  type: "huber"
  huber_delta: 0.5
```

### Data Parameters
```yaml
data:
  input_window: 27
  batch_size: 128
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## 📈 Training Features

- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence
- **Gradient Clipping**: Stabilizes training
- **Checkpointing**: Saves best model automatically
- **TensorBoard Logging**: Real-time training visualization
- **Mixed Precision**: Optional for faster training

## 📊 Evaluation & Backtesting

### Model Evaluation

The system provides comprehensive regression evaluation:

- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **R² Score**: Goodness of fit measure
- **Sign Accuracy**: Directional prediction accuracy
- **MAPE**: Mean Absolute Percentage Error
- **Correlation**: Linear relationship strength
- **Residual Analysis**: Error distribution and patterns

### Backtesting Features

The backtesting engine simulates real futures trading with:

**Trading Mechanics:**
- Leverage support (configurable multiplier)
- Long and short positions
- Maker/taker fee simulation
- Slippage modeling
- Funding rate calculations
- Position pyramiding (up to 3 levels)
- Trailing stop loss
- Dynamic position sizing

**Risk Management:**
- Stop loss and take profit
- Maximum drawdown limits
- Daily loss limits
- Trend reversal detection
- Compound profit reinvestment

**Rich Terminal Output:**
- Color-coded performance metrics (green=profit, red=loss)
- Detailed trade-by-trade breakdown
- Summary tables with key statistics
- Real-time progress tracking
- Professional formatting with tables and panels

## 🎨 Visualizations

Automatically generated plots:

1. **Training Curves**: Training and validation loss over epochs
2. **Predicted vs Actual**: Scatter plot showing prediction quality
3. **Error Distribution**: Histogram of prediction errors
4. **Residual Analysis**: Q-Q plot and residual patterns
5. **Learning Rate Schedule**: LR changes over training

## 🔧 Design Patterns

### Strategy Pattern
The data processing pipeline uses the Strategy Pattern for flexibility:

```python
class DataProcessingStrategy(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pass
```

This allows easy swapping of different preprocessing strategies without modifying core logic.

### Factory Pattern
Data loaders are created using a factory for consistent initialization:

```python
data_factory = DataLoaderFactory(
    data_path=data_path,
    processing_strategy=processing_strategy,
    ...
)
data_loaders = data_factory.create_data_loaders()
```

## 🧪 Code Quality

The project follows strict coding standards:

- **PEP 8 Compliance**: All code formatted with Black
- **Type Hints**: Full type annotations throughout
- **Docstrings**: Google-style documentation
- **Clean Code Principles**: Single responsibility, meaningful names
- **Modular Design**: Clear separation of concerns

## 📝 Usage Examples

### Custom Training

```python
from src.config.config_loader import ConfigLoader
from src.models.mamba_regressor import MambaRegressor

config = ConfigLoader('config.yaml')
model = MambaRegressor(
    input_dim=13,
    d_model=128,
    d_state=16,
    d_conv=4,
    n_layers=4,
    output_dim=1,
)
```

### Evaluation Only

```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model, device)
metrics = evaluator.evaluate(test_loader)
evaluator.print_metrics(metrics)
# Returns: MAE, RMSE, R², Sign Accuracy, etc.
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config.yaml
- Decrease `d_model` or `n_layers`
- Enable gradient checkpointing

### Slow Training
- Increase `batch_size` if memory allows
- Reduce `num_workers` if CPU-bound
- Enable mixed precision training

### Poor Performance
- Increase `n_layers` or `d_model`
- Adjust `learning_rate`
- Check data quality and normalization

## 📚 References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://github.com/state-spaces/mamba)
- [Modular MAX Platform](https://docs.modular.com/max)
- [Mojo Programming Language](https://docs.modular.com/mojo)

## 📄 License

This project is part of the Eunai cryptocurrency prediction system.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ by Aurum**
