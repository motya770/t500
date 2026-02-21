"""Inflation-Stock correlation models.

PyTorch deep learning and scikit-learn boosting models for analyzing
the relationship between inflation rates and stock prices.

PyTorch models:
    - LSTM: Long Short-Term Memory for temporal dependencies
    - GRU: Gated Recurrent Unit (lighter LSTM alternative)
    - Transformer: Self-attention for long-range dependencies
    - TCN: Temporal Convolutional Network with dilated causal convolutions
    - Regime Switching: Learns distinct correlation regimes

Scikit-learn boosting models:
    - Gradient Boosting: Classic gradient boosted trees
    - AdaBoost: Adaptive boosting focused on hard examples
    - Hist Gradient Boosting: Fast histogram-based gradient boosting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Data preparation utilities
# ---------------------------------------------------------------------------


def prepare_lagged_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    n_lags: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create lagged features for tabular models.

    Returns (X, y, feature_names).
    """
    result = df[feature_cols + [target_col]].copy().dropna()
    lag_names = []

    for col in feature_cols:
        for lag in range(1, n_lags + 1):
            name = f"{col}_lag{lag}"
            result[name] = result[col].shift(lag)
            lag_names.append(name)

    # Current values + lagged values
    all_feature_names = list(feature_cols) + lag_names
    result = result.dropna()

    X = result[all_feature_names].values
    y = result[target_col].values
    return X, y, all_feature_names


def prepare_sequences(
    data: np.ndarray,
    seq_length: int,
    target_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sequences for RNN / Transformer / TCN models.

    Args:
        data: array of shape (n_samples, n_features)
        seq_length: lookback window length
        target_idx: column index of the target variable

    Returns:
        X: (n_sequences, seq_length, n_features) tensor
        y: (n_sequences,) tensor
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))


# ---------------------------------------------------------------------------
# PyTorch model definitions
# ---------------------------------------------------------------------------


class LSTMPredictor(nn.Module):
    """LSTM network for time-series regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class GRUPredictor(nn.Module):
    """GRU network for time-series regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TransformerPredictor(nn.Module):
    """Transformer encoder for time-series regression."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


class _TCNBlock(nn.Module):
    """Single residual block for a Temporal Convolutional Network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout(self.relu(self.conv1(x)[..., : x.size(2)]))
        out = self.dropout(self.relu(self.conv2(out)[..., : x.size(2)]))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNPredictor(nn.Module):
    """Temporal Convolutional Network for time-series regression."""

    def __init__(
        self,
        input_dim: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 32]

        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        x = self.network(x)
        return self.fc(x[:, :, -1]).squeeze(-1)


class RegimeSwitchingNet(nn.Module):
    """Mixture-of-experts network that learns distinct correlation regimes.

    A gating network assigns probabilities over *n_regimes* regimes.
    Each regime has its own expert MLP.  The final prediction is a
    probability-weighted combination of expert outputs.
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 3,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes),
            nn.Softmax(dim=-1),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_regimes)
        ])
        self.n_regimes = n_regimes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            inp = x[:, -1, :]
        else:
            inp = x
        regime_probs = self.gate(inp)
        expert_out = torch.stack(
            [expert(inp).squeeze(-1) for expert in self.experts], dim=-1,
        )
        return (expert_out * regime_probs).sum(dim=-1)

    def get_regime_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            inp = x[:, -1, :]
        else:
            inp = x
        return self.gate(inp)


# ---------------------------------------------------------------------------
# PyTorch training loop
# ---------------------------------------------------------------------------


def _train_pytorch_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.001,
    progress_callback=None,
) -> dict:
    """Train a model with early stopping and return history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5,
    )
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 40

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)

        scheduler.step(val_loss.item())
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, epochs, loss.item(), val_loss.item())

    if best_state:
        model.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------


def run_pytorch_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_type: str = "lstm",
    seq_length: int = 3,
    epochs: int = 200,
    lr: float = 0.001,
    hidden_dim: int = 64,
    n_regimes: int = 3,
    progress_callback=None,
) -> dict:
    """Run a PyTorch model for inflation-stock correlation analysis.

    Parameters
    ----------
    df : DataFrame
        Must contain *target_col* and all *feature_cols*, sorted by time.
    target_col : str
        Column to predict (e.g. annual stock return).
    feature_cols : list[str]
        Columns used as input features (e.g. inflation rate).
    model_type : str
        One of ``"lstm"``, ``"gru"``, ``"transformer"``, ``"tcn"``,
        ``"regime_switching"``.
    seq_length : int
        Lookback window (number of time steps).
    epochs, lr, hidden_dim, n_regimes
        Model hyperparameters.
    progress_callback : callable or None
        ``callback(epoch, total_epochs, train_loss, val_loss)``.

    Returns
    -------
    dict with keys: ``model_type``, ``metrics``, ``training_history``,
    ``predictions``, and optionally ``regime_probabilities``.
    """
    all_cols = list(feature_cols) + [target_col]
    data = df[all_cols].dropna()

    if len(data) < seq_length + 4:
        return {
            "error": (
                f"Not enough data. Need at least {seq_length + 4} rows, "
                f"got {len(data)}."
            )
        }

    # ---- scale ----
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_raw = scaler_X.fit_transform(data[feature_cols].values)
    y_raw = scaler_y.fit_transform(data[[target_col]].values).ravel()
    scaled = np.column_stack([X_raw, y_raw])

    target_idx = len(feature_cols)
    X_seq, y_seq = prepare_sequences(scaled, seq_length, target_idx=target_idx)

    if len(X_seq) < 4:
        return {"error": "Not enough sequences. Try a shorter sequence length."}

    # ---- train / val / test split (70 / 15 / 15) ----
    n = len(X_seq)
    train_end = max(2, int(n * 0.70))
    val_end = max(train_end + 1, int(n * 0.85))
    X_train, X_val, X_test = X_seq[:train_end], X_seq[train_end:val_end], X_seq[val_end:]
    y_train, y_val, y_test = y_seq[:train_end], y_seq[train_end:val_end], y_seq[val_end:]

    input_dim = X_seq.shape[2]

    # ---- create model ----
    if model_type == "lstm":
        model = LSTMPredictor(input_dim, hidden_dim=hidden_dim)
    elif model_type == "gru":
        model = GRUPredictor(input_dim, hidden_dim=hidden_dim)
    elif model_type == "transformer":
        nhead = 4 if hidden_dim % 4 == 0 else (2 if hidden_dim % 2 == 0 else 1)
        model = TransformerPredictor(
            input_dim, d_model=hidden_dim, nhead=nhead,
        )
    elif model_type == "tcn":
        model = TCNPredictor(
            input_dim,
            num_channels=[hidden_dim, hidden_dim, max(8, hidden_dim // 2)],
        )
    elif model_type == "regime_switching":
        model = RegimeSwitchingNet(
            input_dim, n_regimes=n_regimes, hidden_dim=hidden_dim,
        )
    else:
        return {"error": f"Unknown model type: {model_type}"}

    # ---- train ----
    history = _train_pytorch_model(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs, lr=lr, progress_callback=progress_callback,
    )

    # ---- evaluate ----
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).numpy()
        val_pred = model(X_val).numpy()
        test_pred = model(X_test).numpy() if len(X_test) > 0 else np.array([])
        all_pred = model(X_seq).numpy()

    inv = lambda a: scaler_y.inverse_transform(a.reshape(-1, 1)).ravel()
    train_pred_orig = inv(train_pred)
    val_pred_orig = inv(val_pred)
    all_pred_orig = inv(all_pred)
    y_train_orig = inv(y_train.numpy())
    y_val_orig = inv(y_val.numpy())
    y_all_orig = inv(y_seq.numpy())

    metrics: dict[str, float] = {
        "train_r2": float(r2_score(y_train_orig, train_pred_orig)) if len(y_train_orig) > 1 else 0.0,
        "train_mse": float(mean_squared_error(y_train_orig, train_pred_orig)),
        "train_mae": float(mean_absolute_error(y_train_orig, train_pred_orig)),
    }
    if len(y_val_orig) > 1:
        metrics["val_r2"] = float(r2_score(y_val_orig, val_pred_orig))
        metrics["val_mse"] = float(mean_squared_error(y_val_orig, val_pred_orig))
        metrics["val_mae"] = float(mean_absolute_error(y_val_orig, val_pred_orig))
    if len(test_pred) > 1:
        test_pred_orig = inv(test_pred)
        y_test_orig = inv(y_test.numpy())
        metrics["test_r2"] = float(r2_score(y_test_orig, test_pred_orig))
        metrics["test_mse"] = float(mean_squared_error(y_test_orig, test_pred_orig))
        metrics["test_mae"] = float(mean_absolute_error(y_test_orig, test_pred_orig))

    # year labels for plotting
    data_index = data.index[seq_length:]
    if "year" in df.columns:
        years = df.loc[data_index, "year"].tolist()
    else:
        years = list(range(len(all_pred_orig)))

    result = {
        "model_type": model_type,
        "metrics": metrics,
        "training_history": history,
        "predictions": {
            "actual": y_all_orig.tolist(),
            "predicted": all_pred_orig.tolist(),
            "years": years,
            "train_size": train_end,
            "val_size": val_end - train_end,
        },
    }

    if model_type == "regime_switching":
        with torch.no_grad():
            regime_probs = model.get_regime_probabilities(X_seq).numpy()
        result["regime_probabilities"] = regime_probs.tolist()

    return result


def run_boosting_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_type: str = "gradient_boosting",
    n_lags: int = 3,
    n_estimators: int = 100,
    cv_folds: int = 5,
) -> dict:
    """Run a scikit-learn boosting model for inflation-stock analysis.

    Parameters
    ----------
    df : DataFrame
        Must contain *target_col* and all *feature_cols*, sorted by time.
    model_type : str
        One of ``"gradient_boosting"``, ``"adaboost"``,
        ``"hist_gradient_boosting"``.
    n_lags : int
        Number of lagged features to generate.
    n_estimators : int
        Number of boosting iterations.
    cv_folds : int
        Time-series cross-validation folds.

    Returns
    -------
    dict with keys: ``model_type``, ``metrics``, ``cv_results``,
    ``predictions``, ``feature_importances``.
    """
    X, y, feature_names = prepare_lagged_features(
        df, target_col, feature_cols, n_lags,
    )

    if len(X) < cv_folds + 2:
        return {
            "error": (
                f"Not enough data. Need at least {cv_folds + 2} rows "
                f"after lagging, got {len(X)}."
            )
        }

    # ---- train / test split (80 / 20) ----
    n = len(X)
    test_start = max(2, int(n * 0.80))
    X_train_all, X_test = X[:test_start], X[test_start:]
    y_train_all, y_test = y[:test_start], y[test_start:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([]).reshape(0, X_train_all.shape[1])
    X_all_scaled = scaler.transform(X)

    # ---- create model ----
    if model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
    elif model_type == "adaboost":
        model = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            random_state=42,
        )
    elif model_type == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        return {"error": f"Unknown model type: {model_type}"}

    # ---- time-series CV on training data only ----
    effective_folds = min(cv_folds, len(X_train_all) - 1)
    tscv = TimeSeriesSplit(n_splits=effective_folds)
    cv_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
        model.fit(X_train_scaled[train_idx], y_train_all[train_idx])
        pred = model.predict(X_train_scaled[val_idx])
        cv_results.append({
            "fold": fold_idx + 1,
            "r2": float(r2_score(y_train_all[val_idx], pred)) if len(val_idx) > 1 else 0.0,
            "mse": float(mean_squared_error(y_train_all[val_idx], pred)),
            "mae": float(mean_absolute_error(y_train_all[val_idx], pred)),
        })

    # ---- final fit on training data, predict everything ----
    model.fit(X_train_scaled, y_train_all)
    train_pred = model.predict(X_train_scaled)
    all_pred = model.predict(X_all_scaled)

    metrics: dict[str, float] = {
        "train_r2": float(r2_score(y_train_all, train_pred)) if len(y_train_all) > 1 else 0.0,
        "train_mse": float(mean_squared_error(y_train_all, train_pred)),
        "train_mae": float(mean_absolute_error(y_train_all, train_pred)),
    }

    cv_r2 = [f["r2"] for f in cv_results]
    cv_mae = [f["mae"] for f in cv_results]
    metrics["cv_mean_r2"] = float(np.mean(cv_r2))
    metrics["cv_mean_mae"] = float(np.mean(cv_mae))

    if len(X_test) > 1:
        test_pred = model.predict(X_test_scaled)
        metrics["test_r2"] = float(r2_score(y_test, test_pred))
        metrics["test_mse"] = float(mean_squared_error(y_test, test_pred))
        metrics["test_mae"] = float(mean_absolute_error(y_test, test_pred))

    importances = (
        model.feature_importances_
        if hasattr(model, "feature_importances_")
        else np.zeros(len(feature_names))
    )
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )

    # year labels
    cleaned = df[feature_cols + [target_col]].dropna()
    valid_idx = cleaned.index[n_lags:]
    if "year" in df.columns:
        years = df.loc[valid_idx, "year"].tolist()
    else:
        years = list(range(len(all_pred)))

    return {
        "model_type": model_type,
        "metrics": metrics,
        "cv_results": cv_results,
        "predictions": {
            "actual": y.tolist(),
            "predicted": all_pred.tolist(),
            "years": years,
            "train_size": test_start,
        },
        "feature_importances": importance_df.to_dict("records"),
        "feature_names": feature_names,
    }
