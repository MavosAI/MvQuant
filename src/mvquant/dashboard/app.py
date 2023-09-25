import numpy as np
import pandas as pd
import os
import torch
from mvquant.dataloader.dataloader import DatasetCustom
from mvquant.dataloader.api import *
from mvquant.models import DLinear, NLinear, PatchTST
from pandas.tseries.offsets import BusinessDay
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import streamlit as st
from types import SimpleNamespace
import click
from pathlib import Path

configs = SimpleNamespace(
    task_name="long_term_forecast",
    batch_size=512,
    is_training=1,
    seq_len=96,
    pred_len=96,
    label_len=0,
    seasonal_patterns=None,  # not used in this dataset
    moving_avg=25,  # 25 default
    embed="timeF",
    activation="gelu",
    output_attention=False,
    freq="D",
    enc_in=1,
    num_class=1,
    individual=True,
    scale=True,
    time_enc=1,
    features="S",
    dropout=0.1,
    # transformer model
    d_model=16,
    e_layers=3,
    d_ff=128,
    c_out=1,
    factor=3,
    n_heads=4,
    head_dropout=0,
    patch_len=16,
    stride=8,
)


@st.cache_resource
def load_model(symbol, model_type, model_path="./src/mvquant/models/checkpoints"):
    model_path = Path(model_path) / f"model_{symbol}_{model_type}.pth"
    models = dict(
        dlinear=DLinear(configs),
        nlinear=NLinear(configs),
        patchtst=PatchTST(configs),
    )
    model = models.get(model_type, DLinear(configs))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    scaler = checkpoint["scaler"]
    return model, scaler


@st.cache_data
def plot_forecast(
    symbol = "VNINDEX",
    model_type = "dlinear",
    template = "plotly_dark",
    data_path="./src/mvquant/datasets",
    holiday_path = "vn_holiday_2025.csv",
    model_path = "./src/mvquant/models/checkpoints"
):
    symbol = symbol.upper()
    data_path = Path(data_path)
    model_type = model_type.lower()
    tz = timezone(timedelta(hours=7))
    data = pd.read_parquet(data_path / f"{symbol}.parquet")
    data.rowDate = pd.to_datetime(data.rowDate, format="%Y-%m-%d")
    df_holiday = pd.read_csv(data_path / holiday_path)
    start_time = int((data.rowDate.max() + BusinessDay(1)).timestamp())
    end_time = int(datetime.now(tz=tz).timestamp())
    actual_df = get_historical_price_vnd(symbol=symbol, start_time=start_time, end_time=end_time)
    model, scaler = load_model(symbol, model_type, model_path)

    test_dataset = DatasetCustom(
        flag="test",
        features=configs.features,
        data_path=data_path / f"{symbol}.parquet",
        target="last_closeRaw",
        timeenc=configs.time_enc,
        freq=configs.freq,
        scale=configs.scale,
        size=(configs.seq_len, configs.label_len, configs.pred_len),
        train_test_ratio=(0.95, 0.05),
        scaler=scaler,
    )
    model.eval()
    with torch.no_grad():
        pred_seq_x, pred_seq_y, pred_seq_x_mark, pred_seq_y_mark = test_dataset[
            len(test_dataset.data_x) - configs.seq_len
        ]
        pred_seq_x = torch.as_tensor(pred_seq_x)[None, :].to(torch.float32)
        pred_seq_y = torch.as_tensor(pred_seq_y)[None, :].to(torch.float32)
        pred_new = model(pred_seq_x, pred_seq_x_mark, pred_seq_y, pred_seq_y_mark)[:, -configs.pred_len :, 0:]
    pi95, pi80 = (0.19913366883993147, 0.14577873051166546)

    last_days = np.round(scaler.inverse_transform(pred_seq_x.numpy().reshape(1, -1))[0], 2)
    forecast_values = np.round(scaler.inverse_transform(pred_new.numpy().reshape(1, -1))[0], 2)
    forecast_values_pi95_low = np.round(forecast_values * (1 - pi95), 2)
    forecast_values_pi95_high = np.round(forecast_values * (1 + pi95), 2)
    forecast_values_pi80_low = np.round(forecast_values * (1 - pi80), 2)
    forecast_values_pi80_high = np.round(forecast_values * (1 + pi80), 2)

    last_date = data.rowDate.max() + BusinessDay(1)
    pred_dates = pd.bdate_range(
        last_date, pd.Timestamp(last_date) + BusinessDay(configs.pred_len + 50)
    )  # add more 50 days due to holiday offline
    pred_dates = list(filter(lambda x: x not in pd.to_datetime(df_holiday["Date"]).tolist(), pred_dates))
    pred_dates = pred_dates[: configs.pred_len]
    data_visual = data.query("rowDate >= '2023/01/01'")

    colors = {
        "train": "rgb(48, 133, 195)",
        "forecast": "rgb(233, 184, 36)",
        "actual": "rgb(33, 156, 144)",
        "fill95": "rgba(255, 217, 61,0.1)",
        "fill80": "rgba(255, 217, 61,0.15)",
    }

    plot_data = []

    plot_data.append(
        go.Scatter(
            x=data_visual.rowDate,
            y=data_visual.last_closeRaw,
            name="Training",
            legendrank=1,
            line=dict(color=colors["train"], width=2),
        )
    )
    plot_data.append(
        go.Scatter(
            x=pred_dates,
            y=forecast_values,
            name="Forecast",
            line=dict(color=colors["forecast"], width=2),
            legendrank=2,
        )
    )
    if len(actual_df) > 0:
        plot_data.append(
            go.Scatter(
                x=actual_df.rowDate,
                y=actual_df.last_closeRaw,
                name="Actual",
                line=dict(color=colors["actual"], width=2),
                legendrank=3,
                mode="lines",
            )
        )

    plot_data.append(
        go.Scatter(
            x=pred_dates,
            y=forecast_values_pi95_high,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    plot_data.append(
        go.Scatter(
            x=pred_dates,
            y=forecast_values_pi95_low,
            name="Predict Interval 95th",
            fill="tonexty",
            fillcolor=colors["fill95"],
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
        )
    )
    plot_data.append(
        go.Scatter(
            x=pred_dates,
            y=forecast_values_pi80_high,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    plot_data.append(
        go.Scatter(
            x=pred_dates,
            y=forecast_values_pi80_low,
            name="Predict Interval 80th",
            fill="tonexty",
            fillcolor=colors["fill80"],
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
        )
    )
    layout = dict(
        title=dict(text=f"{symbol} LONGTERM FORECASTING"),
        hovermode="x unified",
        font_size=13,
        legend=dict(
            borderwidth=2,
        ),
        template=template,
        yaxis=dict(
            title=symbol,
            showgrid=True,
            gridwidth=1,
        ),
        xaxis=dict(
            title="Date",
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(step="all", label="ALL"),
                    ]
                ),
            ),
        ),
    )
    fig = go.Figure(data=plot_data, layout=layout)

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Mavos Forecast")
    st.title("VIET NAM STOCK INDICES FORECASTING")
    st.text("We apply univariate forecasting to predict stock indices!")
    st.markdown(
        'The forecasting implementation of the paper "[Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/pdf/2205.13504.pdf)"'
    )
    if st.button("Clear all cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
    model_type = st.selectbox("Model Type", options=["DLinear"])
    plot_forecast(symbol="VNINDEX", model_type=model_type)
    plot_forecast(symbol="HNXINDEX", model_type=model_type)
    plot_forecast(symbol="UPINDEX", model_type=model_type)


if __name__ == "__main__":
    main()
