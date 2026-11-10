"""AMLGuard — очередь проверки подозрительных операций.  Запуск:  streamlit run app.py"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import load_transactions, build_features, FEATURES
from src.explain import explain_row

ROOT = Path(__file__).parent
st.set_page_config(page_title="AMLGuard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "model.joblib")
    meta = json.load(open(ROOT / "models" / "meta.json", encoding="utf-8"))
    return model, meta


@st.cache_data(show_spinner="Считаю признаки и риск...")
def score_file(path_or_buffer, _model, rates_json):
    rates = json.loads(rates_json)
    df = load_transactions(path_or_buffer)
    F = build_features(df, rates)
    df["risk"] = _model.predict_proba(F[FEATURES])[:, 1]
    return df, F


model, meta = load_model()
m, b = meta["test_metrics"], meta["baseline_metrics"]

st.title("🛡️ AMLGuard")
st.caption("Ранжирование банковских операций по риску отмывания денег для службы финансового мониторинга")

with st.sidebar:
    st.header("Данные")
    up = st.file_uploader("CSV в формате датасета IBM AML", type=["csv"])
    source = up if up is not None else ROOT / "data" / "demo_transactions.csv"
    st.caption("Без загрузки используется демо-файл: 15 минут операций и история тех же счетов за 24 ч.")
    only_new = st.checkbox("Показывать только операции после первых 24 ч файла", value=True,
                           help="Первые 24 ч файла — история для расчёта признаков, а не операции к проверке.")
    share = st.slider("Доля операций на проверку, %", 0.5, 10.0, 1.0, 0.5)

df, F = score_file(source, model, json.dumps(meta["rates"]))
view = df[df["ts"] >= df["ts"].min() + pd.Timedelta(hours=24)] if only_new else df
if view.empty:
    view = df
k = max(1, int(round(len(view) * share / 100)))
queue = view.sort_values("risk", ascending=False).head(k)

tab1, tab2, tab3 = st.tabs(["Очередь проверки", "Карточка операции", "О модели"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Операций в файле к проверке", f"{len(view):,}".replace(",", " "))
    c2.metric("В очереди", f"{len(queue):,}".replace(",", " "))
    if "is_laundering" in view and view["is_laundering"].sum() > 0:
        found = int(queue["is_laundering"].sum()); total = int(view["is_laundering"].sum())
        c3.metric("Найдено отмывания (по разметке)", f"{found} из {total}")
    cols = ["ts", "from_acc", "to_acc", "amount_paid", "payment_currency", "payment_format", "risk"]
    show = queue[cols].rename(columns={"ts": "время", "from_acc": "отправитель", "to_acc": "получатель",
                                       "amount_paid": "сумма", "payment_currency": "валюта",
                                       "payment_format": "способ", "risk": "индекс риска"})
    show["индекс риска"] = (show["индекс риска"] * 100).round(1)
    st.dataframe(show, use_container_width=True)
    st.download_button("Скачать очередь (CSV)", show.to_csv(index=True).encode("utf-8-sig"),
                       "amlguard_queue.csv", "text/csv")
    st.subheader("Сводка")
    s1, s2 = st.columns(2)
    s1.write("Способы оплаты в очереди"); s1.bar_chart(queue["payment_format"].value_counts())
    s2.write("Самые рискованные отправители")
    s2.dataframe(queue.groupby("from_acc")["risk"].agg(["count", "max"]).sort_values("max", ascending=False).head(10)
                 .rename(columns={"count": "операций в очереди", "max": "макс. риск"}))

with tab2:
    if queue.empty:
        st.info("Очередь пуста.")
    else:
        idx = st.selectbox("Операция из очереди", queue.index[:200],
                           format_func=lambda i: f"{df.at[i, 'ts']:%d.%m %H:%M} · {df.at[i, 'from_acc']} → "
                                                 f"{df.at[i, 'to_acc']} · риск {df.at[i, 'risk']*100:.1f}")
        row = df.loc[idx]
        a, b2, c = st.columns(3)
        a.metric("Индекс риска", f"{row['risk']*100:.1f} / 100")
        b2.metric("Сумма", f"{row['amount_paid']:,.2f} {row['payment_currency']}".replace(",", " "))
        c.metric("Способ оплаты", row["payment_format"])
        st.write(f"**Отправитель:** {row['from_acc']}  →  **Получатель:** {row['to_acc']}  ·  {row['ts']:%d.%m.%Y %H:%M}")
        st.subheader("Главные факторы риска")
        st.caption("Вклад = насколько снизится риск, если заменить значение признака типичным.")
        ex = explain_row(model, F.loc[[idx], FEATURES], meta["reference"])
        ex["вклад в риск"] = (ex["вклад в риск"] * 100).round(2)
        st.dataframe(ex, use_container_width=True, hide_index=True)
        st.subheader("История счетов за 24 ч до операции")
        w = df[(df["ts"] < row["ts"]) & (df["ts"] >= row["ts"] - pd.Timedelta(hours=24))]
        h1, h2 = st.columns(2)
        h1.write("Исходящие отправителя"); h1.dataframe(w[w["from_acc"] == row["from_acc"]][["ts", "to_acc", "amount_paid", "payment_currency", "payment_format"]].tail(20))
        h2.write("Входящие получателя"); h2.dataframe(w[w["to_acc"] == row["to_acc"]][["ts", "from_acc", "amount_paid", "payment_currency", "payment_format"]].tail(20))

with tab3:
    st.subheader("Качество на тестовом периоде")
    t = pd.DataFrame({"Модель (бустинг)": m, "Правило-baseline": b}).T
    t = t.rename(columns={"Recall_at_1pct": "Recall@1%"})
    st.dataframe(t.round(3), use_container_width=True)
    st.write(f"Проверяя 1% самых рискованных операций, модель находит **{m['Recall_at_1pct']:.0%}** всех случаев отмывания "
             f"(правило — {b['Recall_at_1pct']:.0%}, случайная проверка — около 1%).")
    st.subheader("Ограничения")
    st.write("Данные синтетические (IBM AML, около 10 дней). Признаки на уровне сети связей не использовались. "
             "В демо-файле история ограничена 24 часами, поэтому риск может немного отличаться от ноутбука.")
