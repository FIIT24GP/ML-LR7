from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy.special import expit

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        """
        Обучает новую базовую модель и добавляет ее в ансамбль.
        """
        # Проверка: если x разреженная матрица, преобразуем в плотную
        if issparse(x):
            x = x.toarray()

        # Генерация бутстрап-выборки
        x_bootstrap, y_bootstrap = resample(x, y, n_samples=int(len(x) * self.subsample))

        # Создаем и обучаем базовую модель на бутстрап-выборке
        base_model = self.base_model_class(**self.base_model_params)
        base_model.fit(x_bootstrap, y_bootstrap)

        # Получаем предсказания для обучающей выборки
        base_preds = base_model.predict(x)

        # Если base_preds разреженная матрица, преобразуем в плотную
        if issparse(base_preds):
            base_preds = base_preds.toarray()

        # Оптимизация гаммы
        gamma = self.find_optimal_gamma(y, predictions, base_preds)

        # Добавляем модель и гамму в соответствующие списки
        self.models.append(base_model)
        self.gammas.append(gamma)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Обучает модель на тренировочном наборе данных и выполняет валидацию на валидационном наборе.
        """
        # Преобразуем разреженные матрицы в плотные (если они разрежены)
        if issparse(x_train):
            x_train = x_train.toarray()

        if issparse(x_valid):
            x_valid = x_valid.toarray()

        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        train_errors = []
        valid_errors = []

        best_valid_error = float('inf')
        rounds_without_improvement = 0

        for _ in range(self.n_estimators):
            # Обучаем новую базовую модель
            self.fit_new_base_model(x_train, y_train, train_predictions)

            # Обновление предсказаний для обучающей и валидационной выборки
            train_predictions += self.learning_rate * self.models[-1].predict(x_train) * self.gammas[-1]
            valid_predictions += self.learning_rate * self.models[-1].predict(x_valid) * self.gammas[-1]

            # Вычисление ошибок
            train_error = self.loss_fn(y_train, train_predictions)
            valid_error = self.loss_fn(y_valid, valid_predictions)

            train_errors.append(train_error)
            valid_errors.append(valid_error)

            # Ранняя остановка
            if self.early_stopping_rounds is not None:
                if valid_error < best_valid_error:
                    best_valid_error = valid_error
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= self.early_stopping_rounds:
                    print(f"Early stopping at round {_}")
                    break

        # Построение графика ошибок, если требуется
        if self.plot:
            self.plot_errors(train_errors, valid_errors)


    def loss_fn(self, y_true, y_pred):
        # Здесь используем логистическую регрессию как пример, можно подставить другую функцию потерь
        return np.mean((y_true - y_pred) ** 2)

    def plot_errors(self, train_errors, val_errors):
        plt.plot(train_errors, label='Train Error')
        plt.plot(val_errors, label='Validation Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

    def predict_proba(self, x):
        """
        Вычисляет вероятности принадлежности классу для каждого образца.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.

        Возвращает
        ----------
        probabilities : array-like, форма (n_samples, n_classes)
            Вероятности для каждого класса.
        """
        # Инициализация финальных предсказаний как нулей
        final_predictions = np.zeros(x.shape[0])

        # Суммируем предсказания всех базовых моделей
        for model, gamma in zip(self.models, self.gammas):
            final_predictions += self.learning_rate * model.predict(x) * gamma

        # Применяем сигмоидальную функцию для получения вероятностей
        probs = expit(final_predictions)  # expit — это сигмоидальная функция

        # Возвращаем вероятности для двух классов (0 и 1)
        return np.column_stack([1 - probs, probs])  # Вероятность для класса 0 и класса 1


    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        """
        Находит оптимальное значение гаммы для минимизации функции потерь.

        Параметры
        ----------
        y : array-like, форма (n_samples,)
            Целевые значения.
        old_predictions : array-like, форма (n_samples,)
            Предыдущие предсказания ансамбля.
        new_predictions : array-like, форма (n_samples,)
            Новые предсказания базовой модели.

        Возвращает
        ----------
        gamma : float
            Оптимальное значение гаммы.

        Примечания
        ----------
        Значение гаммы определяется путем минимизации функции потерь.
        """
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        """
        Возвращает важность признаков в обученной модели.

        Возвращает
        ----------
        importances : array-like, форма (n_features,)
            Важность каждого признака.

        Примечания
        ----------
        Важность признаков определяется по вкладу каждого признака в финальную модель.
        """
        # Инициализация массива для суммарной важности
        total_importances = np.zeros(self.models[0].n_features_in_)

        # Собираем важность признаков для каждой базовой модели
        for model in self.models:
            # У базовой модели есть атрибут `feature_importances_`, если это дерево решений
            if hasattr(model, 'feature_importances_'):
                total_importances += model.feature_importances_

        # Нормализуем важность признаков
        total_importances /= len(self.models)

        return total_importances
