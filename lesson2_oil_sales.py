import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# -----------------------------------------------------------------------------
# 0) Reproducibility / Воспроизводимость
# -----------------------------------------------------------------------------
# Фиксируем генератор случайных чисел, чтобы при каждом запуске получать
# одинаковый synthetic dataset и сопоставимые метрики.
# Fix random generator so each run produces the same synthetic dataset and metrics.
rng = np.random.default_rng(42)

# -----------------------------------------------------------------------------
# 1) Synthetic reference data / Синтетические исходные данные
# -----------------------------------------------------------------------------
# Справочники категориальных признаков:
# - brands: марка авто
# - oils: тип вязкости масла
# - months: номер месяца (1..12)
# Categorical domains:
# - brands: car brand
# - oils: oil viscosity type
# - months: month number (1..12)
brands = ["Toyota", "BMW", "Audi", "Mercedes", "Ford", "Honda", "Nissan", "Kia", "Hyundai", "Volkswagen"]
oils = ["0W-20", "5W-30", "5W-40", "10W-40"]
months = np.arange(1, 13)

# Сюда будем складывать все сгенерированные строки (наблюдения).
# All generated rows will be accumulated here.
rows = []

# Генерируем данные за 3 года.
# We generate data for 3 years.
for year in [2023, 2024, 2025]:
    for month in months:
        # Для каждого месяца создаем много "чеков"/наблюдений.
        # Это имитирует продажи в разных магазинах/днях в пределах месяца.
        # For each month we create many observations, simulating store/day sales.
        for _ in range(80):
            # Случайно выбираем марку авто и тип масла.
            # Randomly choose car brand and oil type.
            brand = rng.choice(brands)
            oil = rng.choice(oils)

            # Сезонные индикаторы:
            # winter=1 для декабря, января, февраля
            # summer=1 для июня, июля, августа
            # Season flags:
            # winter=1 for Dec/Jan/Feb, summer=1 for Jun/Jul/Aug.
            winter = 1 if month in [12, 1, 2] else 0
            summer = 1 if month in [6, 7, 8] else 0

            # Базовая цена зависит от вязкости, затем добавляем шум.
            # Base price depends on oil viscosity, then we add random noise.
            base_price = {"0W-20": 52, "5W-30": 48, "5W-40": 50, "10W-40": 44}[oil]
            price = base_price + rng.normal(0, 2.5)

            # Скидка тоже синтетическая:
            # нормальное распределение вокруг 8%, ограничение в диапазон [0%, 25%].
            # Discount is synthetic too:
            # normal distribution around 8%, clipped to [0%, 25%].
            discount = np.clip(rng.normal(0.08, 0.05), 0, 0.25)  # 0..25%

            # Популярность бренда как коэффициент спроса.
            # >1.0 повышает спрос, <1.0 снижает.
            # Brand popularity demand multiplier.
            brand_factor = {
                "Toyota": 1.10, "BMW": 0.95, "Audi": 0.92, "Mercedes": 0.90,
                "Ford": 1.00, "Honda": 1.05, "Nissan": 1.03, "Kia": 1.08,
                "Hyundai": 1.07, "Volkswagen": 0.98
            }[brand]

            # Для каждого масла задаем сезонные бонусы:
            # - winter bonus: как ведет себя спрос зимой
            # - summer bonus: как ведет себя спрос летом
            # Oil-specific seasonality multipliers.
            oil_winter_bonus = {"0W-20": 1.08, "5W-30": 1.10, "5W-40": 1.02, "10W-40": 0.95}[oil]
            oil_summer_bonus = {"0W-20": 0.98, "5W-30": 1.00, "5W-40": 1.05, "10W-40": 1.07}[oil]

            # Начинаем с нейтрального множителя и применяем сезонные поправки.
            # Start from neutral season factor and apply season multipliers.
            season_factor = 1.0
            if winter:
                season_factor *= oil_winter_bonus
            if summer:
                season_factor *= oil_summer_bonus

            # Формула спроса (синтетическая, но "похожая на реальность"):
            # - Базовый уровень 120 единиц
            # - Множители бренда и сезона
            # - Скидка увеличивает спрос (коэфф. 1 + 1.3*discount)
            # - Цена относительно базовой слегка снижает/повышает спрос
            # - Добавляем случайный шум, чтобы данные были неидеальными
            # Synthetic demand formula:
            # - baseline 120 units
            # - brand and season multipliers
            # - discount increases demand
            # - higher-than-base price slightly reduces demand
            # - random noise for realism
            demand = (
                120
                * brand_factor
                * season_factor
                * (1 + 1.3 * discount)
                * (1 - 0.012 * (price - base_price))
                + rng.normal(0, 8)
            )

            # Не допускаем слишком маленьких/отрицательных продаж.
            # Prevent unrealistically low/negative sales.
            units_sold = max(5, demand)

            # Округляем некоторые поля для "человеческого" табличного вида.
            # Round fields to make table values more human-readable.
            rows.append([year, month, brand, oil, round(price, 2), round(discount, 3), round(units_sold, 1)])

# Создаем DataFrame для дальнейшего ML-процесса.
# Build DataFrame used by the ML pipeline.
df = pd.DataFrame(rows, columns=[
    "year", "month", "car_brand", "oil_viscosity", "price", "discount", "units_sold"
])

# -----------------------------------------------------------------------------
# 2) Time-based split / Разделение по времени (критически важно)
# -----------------------------------------------------------------------------
# В задачах прогноза нельзя случайно перемешивать данные как в классических
# tabular-задачах: это ведет к data leakage.
# Поэтому:
# - train: до сентября 2025 включительно
# - test:  сентябрь-декабрь 2025 (на "будущем")
# For forecasting tasks we avoid random split to prevent leakage.
# Train uses historical period, test uses future period.
train_df = df[(df["year"] < 2025) | ((df["year"] == 2025) & (df["month"] <= 8))].copy()
test_df = df[(df["year"] == 2025) & (df["month"] > 8)].copy()

# Отделяем признаки (X) от целевой переменной (y).
# Split into features (X) and target (y).
X_train = train_df.drop(columns=["units_sold"])
y_train = train_df["units_sold"]

X_test = test_df.drop(columns=["units_sold"])
y_test = test_df["units_sold"]

# -----------------------------------------------------------------------------
# 3) Preprocessing + Model / Препроцессинг + модель
# -----------------------------------------------------------------------------
# Определяем типы признаков:
# - категориальные: марка авто, вязкость
# - числовые: год, месяц, цена, скидка
# Define feature groups:
# - categorical: brand, viscosity
# - numeric: year, month, price, discount
categorical_features = ["car_brand", "oil_viscosity"]
numeric_features = ["year", "month", "price", "discount"]

# ColumnTransformer:
# - OneHotEncoder для категориальных (модель не понимает строки напрямую)
# - passthrough для числовых (передаем как есть)
# OneHotEncoder(handle_unknown="ignore") нужен для устойчивости:
# если в будущем появится новая категория, код не упадет.
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# RandomForestRegressor:
# - нелинейная модель, хорошо работает на смешанных признаках
# - n_estimators=300: достаточное число деревьев для стабильности
# - n_jobs=-1: использовать все CPU-ядра
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# Pipeline связывает препроцессинг и модель в единый объект:
# fit/predict вызываются в правильном порядке автоматически.
# Pipeline links preprocessing + model into one reproducible workflow.
pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# Обучение на train и предсказание на test.
# Train on historical data, predict on future holdout period.
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

# MAE: средняя абсолютная ошибка в "единицах продаж" (интерпретируемо для бизнеса).
# MAPE: средняя относительная ошибка (доля/процент).
# MAE = business-friendly absolute units error.
# MAPE = relative percentage-like error.
mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.4f}")

# -----------------------------------------------------------------------------
# 4) Future scenario forecast / Сценарный прогноз на будущее
# -----------------------------------------------------------------------------
# Здесь вручную задаем гипотетические кейсы (что-если сценарии):
# - разные месяцы/бренды/вязкости/цены/скидки
# Модель возвращает ожидаемый объем продаж для каждого сценария.
# We create hypothetical what-if cases and estimate expected sales.
future = pd.DataFrame([
    {"year": 2026, "month": 1, "car_brand": "Toyota", "oil_viscosity": "5W-30", "price": 49.0, "discount": 0.10},
    {"year": 2026, "month": 1, "car_brand": "BMW", "oil_viscosity": "0W-20", "price": 54.0, "discount": 0.06},
    {"year": 2026, "month": 7, "car_brand": "Kia", "oil_viscosity": "10W-40", "price": 43.5, "discount": 0.12},
])

future_pred = pipe.predict(future)
future["predicted_units_sold"] = np.round(future_pred, 1)

print("\nFuture predictions:")
print(future)