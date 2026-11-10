"""Объяснение риска отдельной операции и русские названия признаков."""
import numpy as np
import pandas as pd

NAMES_RU = {
    "log_usd": "сумма (лог, USD)", "is_conversion": "есть конвертация валют",
    "same_bank": "один банк у сторон", "self_transfer": "перевод самому себе",
    "round_amount": "круглая сумма", "hour": "час операции", "weekday": "день недели",
    "night": "ночная операция", "snd_cnt_24h": "операций отправителя за 24 ч",
    "snd_sum_24h_log": "сумма отправителя за 24 ч (лог)",
    "snd_hours_since_prev": "часов с прошлой операции отправителя",
    "snd_amount_vs_mean": "сумма / средняя сумма отправителя",
    "snd_uniq_receivers": "уникальных получателей у отправителя",
    "rcv_cnt_24h": "входящих у получателя за 24 ч", "pair_first": "первый перевод этой паре",
    "reverse_seen": "был обратный перевод", "payment_format": "способ оплаты",
    "payment_currency": "валюта платежа",
}


def reference_values(X: pd.DataFrame) -> dict:
    """Типичные значения признаков (медиана / самое частое) по обучающей выборке."""
    ref = {}
    for c in X.columns:
        ref[c] = float(np.nanmedian(X[c])) if pd.api.types.is_numeric_dtype(X[c]) else str(X[c].mode().iloc[0])
    return ref


def explain_row(model, row: pd.DataFrame, ref: dict, top: int = 5) -> pd.DataFrame:
    """Вклад признаков в риск одной операции.

    Для каждого признака значение заменяется типичным; вклад = на сколько
    при этом падает риск. Положительный вклад — признак повышает риск.
    """
    base = model.predict_proba(row)[:, 1][0]
    variants = []
    for c in row.columns:
        r = row.copy()
        r[c] = ref[c]
        variants.append(r)
    alt = model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    out = pd.DataFrame({"признак": [NAMES_RU.get(c, c) for c in row.columns],
                        "значение": [row[c].iloc[0] for c in row.columns],
                        "вклад в риск": base - alt})
    return out.sort_values("вклад в риск", ascending=False).head(top).reset_index(drop=True)
