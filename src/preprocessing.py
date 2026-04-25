import os
import pandas as pd


def load_data(data_path: str = '../data/raw/spotify-tracks-dataset.csv') -> pd.DataFrame:
    """
    Подругажет и возвращает датасета по заданному пути.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError("Проверьте заданный путь")
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка данных: удаление дубликатов, обработка типов данных.

    Функция принимает датафрейм, удаляет в нём дубликаты, удаляет при необходимости пропуски
    и преобразовывает столбец о непристойности композиции к целочисленному типу
    """
    df = df.drop_duplicates()
    if 'explicit' in df.columns:
        df['explicit'] = df['explicit'].astype(int)
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Найдено {missing_count} пропусков")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция создаёт два новых признака: конвертирует длительность трека в минуты
    и признак характеризующий популярность трека
    """
    df['duration_min'] = df['duration_ms'] / 60000
    df['is_popular'] = (df['popularity'] > 70).astype(int)
    return df


def get_audio_features() -> list:
    """
    Возвращает список всех аудио-фич датасета
    """
    return [
        'danceability',
        'energy',
        'speechiness',
        'acousticness',
        'loudness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo'
    ]


def save_processed_data(
    df: pd.DataFrame,
    path: str = '../data/processed/spotify_processed.csv'
) -> None:
    """
    Сохраняет обработанные данные в CSV файл по указанному пути.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("Данные успешно сохранены по пути: {path}")
