import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Анализ", layout="centered")
st.title("📊 Анализ данных по найденным билетам")

st.markdown("✅ Здесь представлены графики по результатам поиска билетов.")

# 🧩 Предположим, что пользователь загрузил данные с главной страницы
# 👉 Альтернативно: можно позволить загрузить CSV вручную
uploaded_file = st.file_uploader("📎 Загрузите CSV-файл с результатами билетов", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Приводим дату к нужному формату
    if "Дата" in df.columns:
        df["Дата"] = pd.to_datetime(df["Дата"])

    # 1. 📈 Средняя цена по дате
    st.markdown("### 📅 Средняя цена по дате вылета")
    fig1, ax1 = plt.subplots()
    df.groupby("Дата")["Цена (₸)"].mean().plot(kind="line", marker="o", ax=ax1)
    ax1.set_ylabel("Цена (₸)")
    ax1.set_xlabel("Дата")
    ax1.set_title("📈 Средняя цена по дате")
    st.pyplot(fig1)

    # 2. ✈️ Частота авиакомпаний
    st.markdown("### ✈️ Частота авиакомпаний")
    fig2, ax2 = plt.subplots()
    df["Авиакомпания"].value_counts().plot(kind="bar", color="skyblue", ax=ax2)
    ax2.set_ylabel("Количество рейсов")
    ax2.set_title("✈️ Авиакомпании в выборке")
    st.pyplot(fig2)

    # 3. 🔁 Количество пересадок
    st.markdown("### 🔁 Количество пересадок")
    fig3, ax3 = plt.subplots()
    df["Пересадок"].value_counts().sort_index().plot(kind="bar", color="orange", ax=ax3)
    ax3.set_ylabel("Количество")
    ax3.set_title("🔁 Пересадки")
    st.pyplot(fig3)

    st.markdown("✅ Анализ завершён.")
else:
    st.info("🔄 Загрузите файл CSV, экспортированный с главной страницы.")
