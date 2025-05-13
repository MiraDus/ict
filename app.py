# flight_app/app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import os

LOG_PATH = "user_activity_log.csv"

def log_action(action, user_id="unknown"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{user_id},{action}\n")


# 🧭 Настройки страницы — САМАЯ ПЕРВАЯ КОМАНДА!
st.set_page_config(page_title="Билеты", layout="wide")

# 🎨 Кастомный CSS-стиль и фон
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1504198453319-5ce911bafcde');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2em;
        border-radius: 15px;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    </style>
""", unsafe_allow_html=True)

# 🏷️ Заголовок и иконка
st.title("🧭 Поиск и рекомендация авиабилетов")
st.markdown("#### 🔍 Найдём лучшие предложения по дате, цене и авиакомпании")

# 🔐 Токен и модель
API_TOKEN = "1b910f874307a16b139978cd28c69972"
model = load_model("flight_model.h5")
le_dep = joblib.load("le_dep.pkl")
le_arr = joblib.load("le_arr.pkl")
le_airline = joblib.load("le_airline.pkl")

if "username" not in st.session_state:
    st.session_state.username = st.text_input("Введите своё имя или никнейм для учёта активности")

if st.session_state.username:
    log_action("🟢 Пользователь зашел на сайт", st.session_state.username)

# 🎛️ Форма в колонках
col1, col2 = st.columns(2)
with col1:
    origin = st.text_input("🛫 Город вылета (IATA)", "ALA")
    depart_date = st.date_input("📅 Дата вылета")
with col2:
    destination = st.text_input("🛬 Город назначения (IATA)", "IST")
    max_price = st.slider("💰 Максимальная цена (₸)", 30000, 300000, 150000)

selected_airline = st.text_input("✈️ Фильтр по авиакомпании (напр. KC)", "").upper()
max_transfers = st.selectbox("🔁 Макс. пересадок", ["Любое", 0, 1, 2])

# 🔘 Поиск и результаты
if st.button("🔍 Найти билеты"):
    log_action(f"🔍 Поиск: {origin} → {destination}, до {max_price}₸", st.session_state.username)
    base_url = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
    collected = []
    current_time = datetime.now().strftime("%d.%m.%Y %H:%M")

    for delta in range(-5, 6):
        date = (depart_date + timedelta(days=delta)).strftime("%Y-%m-%d")
        params = {
            "origin": origin,
            "destination": destination,
            "departure_at": date,
            "currency": "kzt",
            "sorting": "price",
            "limit": 10,
            "token": API_TOKEN
        }

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                for item in response.json().get("data", []):
                    if item.get("price", 0) <= max_price:
                        if selected_airline and item.get("airline", "").upper() != selected_airline:
                            continue
                        if max_transfers != "Любое" and item.get("transfers", 0) > int(max_transfers):
                            continue
                        collected.append({
                            "Дата": item.get("departure_at", "").split("T")[0],
                            "Цена (₸)": item.get("price", 0),
                            "Авиакомпания": item.get("airline", "—").upper(),
                            "Пересадок": item.get("transfers", 0),
                            "Рейс": item.get("flight_number", "—"),
                            "Ссылка": f"https://www.aviasales.kz{item.get('link', '')}"
                        })
        except:
            pass

    if collected:
        df = pd.DataFrame(collected).sort_values("Цена (₸)").head(10)
        st.success(f"✅ Найдено {len(df)} лучших билетов (обновлено: {current_time})")
        st.dataframe(df)

        # 🔥 Горящие билеты
        cheap = df[df["Цена (₸)"] < 50000]
        if not cheap.empty:
            st.warning("🔥 Найдены горящие билеты!")
            st.dataframe(cheap)

        # 🧠 Рекомендация от нейросети
        st.markdown("### 🎯 Рекомендованный билет от нейросети")
        try:
            unknown_airlines = [air for air in df["Авиакомпания"].unique() if air not in le_airline.classes_]
            unknown_dep = [origin] if origin not in le_dep.classes_ else []
            unknown_arr = [destination] if destination not in le_arr.classes_ else []

            if unknown_airlines or unknown_dep or unknown_arr:
                best = df.iloc[0]
            else:
                X_pred = np.array([
                    [
                        le_dep.transform([origin])[0],
                        le_arr.transform([destination])[0],
                        le_airline.transform([row["Авиакомпания"]])[0]
                    ] for _, row in df.iterrows()
                ])
                preds = model.predict(X_pred)
                best_index = np.argmax(preds, axis=0)[0]
                best = df.iloc[best_index]

            st.markdown(f"**{origin} → {destination}, {best['Дата']}**")
            st.markdown(f"💸 **Цена:** {best['Цена (₸)']} ₸")
            st.markdown(f"✈️ **Авиакомпания:** {best['Авиакомпания']}")
            st.markdown(f"[🛒 Купить билет]({best['Ссылка']})", unsafe_allow_html=True)
        except:
            st.info("⚠️ Рекомендация временно недоступна.")
    else:
        st.warning("❌ Билеты не найдены по заданным условиям.")
with st.expander("📌 Часто задаваемые вопросы"):
    st.markdown("""
    **Как найти билеты?**  
    Укажите IATA-коды городов, выберите дату и нажмите "🔍 Найти билеты".

    **Что означает 'рекомендация от нейросети'?**  
    Это билет, который предсказывает модель как наиболее оптимальный по соотношению факторов.

    **Что делать, если билеты не найдены?**  
    Попробуйте изменить дату, увеличить цену или убрать фильтры.

    **Откуда берутся данные?**  
    Используется API TravelPayouts — реальные данные билетов.
    """)

st.markdown("""
    <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1534854638093-bada1813ca19');
            background-size: cover;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255,255,255,0.9);  /* прозрачность */
            padding: 2rem;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)
with st.expander("📊 Посмотреть активность пользователей (только для разработчика)"):
    if os.path.exists(LOG_PATH):
        logs = pd.read_csv(LOG_PATH, names=["Время", "Пользователь", "Действие"], encoding="utf-8")
        st.dataframe(logs.tail(20))
    else:
        st.info("Лог-файл пока пуст.")
