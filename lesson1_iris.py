# --- Импорты: подключаем готовые «кирпичи» из scikit-learn ---
# load_iris — встроенный toy dataset (игрушечный набор): 150 цветков ириса, 4 признака, 3 класса вида.
from sklearn.datasets import load_iris

# train_test_split — разбиение на обучающую и тестовую выборки (train / test split).
from sklearn.model_selection import train_test_split

# Pipeline — конвейер: несколько шагов подряд (препроцессинг → модель) в одном объекте.
from sklearn.pipeline import Pipeline

# StandardScaler — стандартизация признаков: среднее 0, дисперсия 1 (важно для многих линейных моделей).
from sklearn.preprocessing import StandardScaler

# LogisticRegression — логистическая регрессия: классификация (здесь мультикласс: 3 вида ириса).
from sklearn.linear_model import LogisticRegression

# Метрики: accuracy — доля верных ответов; classification_report — precision, recall, F1 по классам.
from sklearn.metrics import accuracy_score, classification_report

# --- Загрузка данных (dataset) ---
# load_iris() возвращает объект Bunch: в нём есть .data (матрица признаков X), .target (метки y), имена и т.д.
# Пример: одна строка X — один цветок; столбцы — длина/ширина чашелистика и лепестка в см.
data = load_iris()

# X (матрица признаков, feature matrix): форма обычно (150, 4) — 150 объектов, 4 числовых признака.
# y (вектор целевой переменной, target): целые числа 0, 1, 2 — номера класса (вида ириса).
X, y = data.data, data.target

# --- Разбиение выборки (split) ---
# Учимся на train, честно проверяемся на test — так оцениваем обобщение, а не «заучивание» train.
# test_size=0.2 — 20% данных в тест, 80% в обучение.
# random_state=42 — фиксируем случайность: при повторном запуске то же самое разбиение (воспроизводимость).
# stratify=y — пропорции классов в train и test как в исходных данных (важно при малом датасете).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Пайплайн (Pipeline): сначала масштабирование, потом модель ---
# Список пар (имя_шага, объект): при fit сначала scaler учится на X_train, потом model получает уже масштабированные X.
# При predict на X_test те же преобразования применяются автоматически — меньше риска «утечки» данных (data leakage).
#
# Математика шага «scaler» (StandardScaler), по столбцу признака j (например, длина чашелистика):
#   μ_j = (1/n) * Σ_i x_i,j          — среднее только по обучающей выборке (train);
#   σ_j = sqrt((1/n) * Σ_i (x_i,j - μ_j)²)   — стандартное отклонение по train (в sklearn — «population», ddof=0);
#   z_i,j = (x_i,j - μ_j) / σ_j      — новое значение признака: нулевое среднее, единичная дисперсия по столбцу на train.
#   Пример: если в см «всё большое», а весах модели мелкие числа, масштабирование выравнивает вклад признаков.
#
# Математика шага «model» (LogisticRegression, мультикласс, K=3 вида ириса):
#   После scaler вектор признаков объекта обозначим x′ ∈ R^d (d=4).
#   Для каждого класса k задаём линейный скор (score): s_k(x′) = w_k^T x′ + b_k
#     (w_k — веса, b_k — смещение / bias; при обучении они подбираются из данных).
#   Вариант «softmax» (multinomial): вероятность класса k
#     P(y=k | x′) = exp(s_k) / Σ_{c=1..K} exp(s_c)   — это softmax(s_1,…,s_K).
#   Предсказание класса: argmax_k P(y=k|x′) (часто то же, что argmax_k s_k).
#   Для двух классов softmax вырождается в сигмоиду: P(y=1|x) = 1 / (1 + exp(-(w^T x + b))).
#   Обучение: подбирают {w_k, b_k}, минимизируя регуляризованную логистическую потерю (cross-entropy + штраф L2 по умолчанию).
#
pipe = Pipeline([
    ("scaler", StandardScaler()),  # шаг 1: z = (x - μ_train) / σ_train по каждому признаку (см. формулы выше).
    ("model", LogisticRegression(max_iter=1000))  # шаг 2: линейные скоры → softmax/ovr; max_iter — лимит итераций солвера.
])

# --- Обучение (fit) ---
# Модель подбирает веса по X_train и y_train: минимизирует функцию потерь (loss) логистической регрессии.
pipe.fit(X_train, y_train)

# --- Оценка на отложенной выборке (inference / evaluation) ---
# predict — для каждого объекта из X_test выдаёт предсказанный класс (0, 1 или 2).
pred = pipe.predict(X_test)

# accuracy — доля совпадений pred с y_test; на ирисе часто высокая, но на несбалансированных классах смотрят ещё F1.
acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc:.4f}")
print("\nClassification report:")
# classification_report: по каждому классу — precision (точность среди предсказанных «да»), recall (полнота),
# f1-score — гармоническое среднее; support — сколько реальных примеров класса было в y_test.
print(classification_report(y_test, pred, target_names=data.target_names))
