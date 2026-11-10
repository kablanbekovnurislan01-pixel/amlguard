"""Загрузка данных IBM AML и расчёт признаков AMLGuard.

Главное правило: все признаки истории считаются только по операциям,
которые были СТРОГО раньше текущей (операции с тем же временем не видны).
"""
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["ts", "from_bank", "from_account", "to_bank", "to_account",
           "amount_received", "receiving_currency", "amount_paid",
           "payment_currency", "payment_format", "is_laundering"]

NUM_FEATURES = [
    "log_usd", "is_conversion", "same_bank", "self_transfer", "round_amount",
    "hour", "weekday", "night",
    "snd_cnt_24h", "snd_sum_24h_log", "snd_hours_since_prev", "snd_amount_vs_mean",
    "snd_uniq_receivers", "rcv_cnt_24h",
    "pair_first", "reverse_seen",
]
CAT_FEATURES = ["payment_format", "payment_currency"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def load_transactions(path, days=None) -> pd.DataFrame:
    """Читает CSV в формате датасета, даёт понятные имена столбцов.

    days — если задано, берутся только первые N суток данных (для скорости).
    """
    df = pd.read_csv(Path(path), header=0, dtype={1: str, 2: str, 3: str, 4: str})
    df = df.iloc[:, :11]
    df.columns = COLUMNS[: df.shape[1]]
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    df = df[df["ts"] < pd.Timestamp("2022-09-11")]  # tail 11-18 Sep: generator artifact
    if days is not None:
        df = df[df["ts"] < df["ts"].min().normalize() + pd.Timedelta(days=days)]
    for c in ["receiving_currency", "payment_currency", "payment_format"]:
        df[c] = df[c].astype(str)
    df["from_acc"] = df["from_bank"].astype(str) + "_" + df["from_account"].astype(str)
    df["to_acc"] = df["to_bank"].astype(str) + "_" + df["to_account"].astype(str)
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    return df


def usd_rates(df: pd.DataFrame) -> dict:
    """Курс валюты к USD из самих данных: медиана отношения сумм в операциях с конвертацией в/из USD."""
    rates = {"US Dollar": 1.0}
    to_usd = df[(df.receiving_currency == "US Dollar") & (df.payment_currency != "US Dollar")]
    r1 = (to_usd.amount_received / to_usd.amount_paid).groupby(to_usd.payment_currency).median()
    from_usd = df[(df.payment_currency == "US Dollar") & (df.receiving_currency != "US Dollar")]
    r2 = (from_usd.amount_paid / from_usd.amount_received).groupby(from_usd.receiving_currency).median()
    for cur, v in pd.concat([r2, r1]).items():
        if np.isfinite(v) and v > 0:
            rates[cur] = float(v)
    # валюты без прямых пар с USD: через валюты с уже известным курсом
    for _ in range(3):
        conv = df[df.payment_currency != df.receiving_currency]
        for cur in set(df.payment_currency) | set(df.receiving_currency):
            if cur in rates:
                continue
            a = conv[(conv.payment_currency == cur) & conv.receiving_currency.isin(rates.keys())]
            if len(a):
                v = (a.amount_received * a.receiving_currency.map(rates) / a.amount_paid).median()
                if np.isfinite(v) and v > 0:
                    rates[cur] = float(v)
    return rates


def _rolling_24h(df: pd.DataFrame, key: str, col: str):
    """Число и сумма операций счёта за 24 ч СТРОГО до текущего момента."""
    order = df.sort_values([key, "ts"], kind="mergesort").index
    s = df.loc[order, [key, "ts", col]].set_index("ts")
    r = s.groupby(key, sort=False)[col].rolling("24h", closed="left")
    cnt = r.count().fillna(0).to_numpy()
    sm = r.sum().fillna(0).to_numpy()
    out_cnt = pd.Series(cnt, index=order).reindex(df.index)
    out_sum = pd.Series(sm, index=order).reindex(df.index)
    return out_cnt, out_sum


def build_features(df: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """Возвращает таблицу признаков (те же строки и порядок, что у df)."""
    df = df.sort_values("ts", kind="mergesort")
    f = pd.DataFrame(index=df.index)
    rate = df["payment_currency"].map(rates).fillna(np.nan)
    usd = df["amount_paid"] * rate
    f["log_usd"] = np.log1p(usd)
    f["is_conversion"] = (df.payment_currency != df.receiving_currency).astype(int)
    f["same_bank"] = (df.from_bank.astype(str) == df.to_bank.astype(str)).astype(int)
    f["self_transfer"] = (df.from_acc == df.to_acc).astype(int)
    f["round_amount"] = ((df.amount_paid % 100) == 0).astype(int)
    f["hour"] = df.ts.dt.hour
    f["weekday"] = df.ts.dt.dayofweek
    f["night"] = (f["hour"] < 6).astype(int)

    tmp = df[["ts", "from_acc", "to_acc"]].assign(usd=usd.fillna(0))
    f["snd_cnt_24h"], snd_sum = _rolling_24h(tmp, "from_acc", "usd")
    f["snd_sum_24h_log"] = np.log1p(snd_sum)
    f["rcv_cnt_24h"], _ = _rolling_24h(tmp, "to_acc", "usd")

    # время с последней операции отправителя, строго раньше текущей
    g = tmp.groupby("from_acc", sort=False)
    first_at_ts = ~tmp.duplicated(["from_acc", "ts"])
    prev_ts = tmp["ts"].where(first_at_ts)
    prev_ts = prev_ts.groupby(tmp["from_acc"]).shift(1)
    prev_ts = prev_ts.groupby([tmp["from_acc"]]).ffill()
    prev_ts = prev_ts.where(first_at_ts).groupby([tmp["from_acc"], tmp["ts"]]).transform("first")
    f["snd_hours_since_prev"] = (tmp["ts"] - prev_ts).dt.total_seconds() / 3600

    # отношение суммы к средней сумме прошлых операций отправителя
    # (операции с тем же временем считаются как «одновременные» и не учитываются)
    by_ts = tmp.groupby(["from_acc", "ts"], sort=False)["usd"].agg(["sum", "count"])
    by_ts = by_ts.sort_index(level=["from_acc", "ts"])
    cs = by_ts.groupby(level=0)[["sum", "count"]].cumsum() - by_ts[["sum", "count"]]
    prev_mean = (cs["sum"] / cs["count"]).replace([np.inf, -np.inf], np.nan)
    pm = prev_mean.reindex(pd.MultiIndex.from_arrays([tmp["from_acc"], tmp["ts"]])).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = usd.to_numpy() / pm
    f["snd_amount_vs_mean"] = pd.Series(ratio, index=tmp.index).replace([np.inf, -np.inf], np.nan).clip(upper=1000)

    # пары счетов: первое время пары, первый ли перевод, был ли обратный перевод раньше
    pair = tmp["from_acc"] + ">" + tmp["to_acc"]
    rev = tmp["to_acc"] + ">" + tmp["from_acc"]
    first_time = tmp.groupby(pair)["ts"].min()
    f["pair_first"] = (tmp["ts"] == pair.map(first_time)).astype(int)
    rev_first = rev.map(first_time)
    f["reverse_seen"] = (rev_first < tmp["ts"]).astype(int)

    # число уникальных получателей отправителя до текущего момента
    pf = pd.Series((tmp["ts"] == pair.map(first_time)) & ~pair.duplicated(), index=tmp.index)
    ev = pd.DataFrame({"acc": tmp["from_acc"], "ts": tmp["ts"], "new": pf.astype(int)})
    per_ts = ev.groupby(["acc", "ts"], sort=True)["new"].sum()
    before = per_ts.groupby(level=0).cumsum() - per_ts
    f["snd_uniq_receivers"] = before.reindex(pd.MultiIndex.from_arrays([ev["acc"], ev["ts"]])).to_numpy()

    for c in CAT_FEATURES:
        f[c] = df[c].astype(str)
    return f.loc[df.index].reindex(df.index)
