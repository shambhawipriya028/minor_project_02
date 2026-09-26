"""Hybrid movie recommendation project using MovieLens latest-small."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "ml-latest-small"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RANDOM_SEED = 42


def ensure_dataset() -> tuple[Path, Path]:
    """Download the small official MovieLens archive if CSVs are not present."""
    if MOVIES_PATH.exists() and RATINGS_PATH.exists():
        return MOVIES_PATH, RATINGS_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading MovieLens latest-small from GroupLens...")
    with urlopen(DATASET_URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        movies_bytes = archive.read("ml-latest-small/movies.csv")
        ratings_bytes = archive.read("ml-latest-small/ratings.csv")
        readme_name = "ml-latest-small/README.txt"
        readme_bytes = archive.read(readme_name) if readme_name in archive.namelist() else b""
    MOVIES_PATH.write_bytes(movies_bytes)
    RATINGS_PATH.write_bytes(ratings_bytes)
    if readme_bytes:
        (DATA_DIR / "README.txt").write_bytes(readme_bytes)
    return MOVIES_PATH, RATINGS_PATH


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies_path, ratings_path = ensure_dataset()
    movies = pd.read_csv(movies_path, encoding="utf-8")
    ratings = pd.read_csv(ratings_path, encoding="utf-8")
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    movies["movieId"] = movies["movieId"].astype(int)
    return movies, ratings


class MovieRecommendationSystem:
    """Content-based, user-user collaborative, and hybrid recommendations."""

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame, neighbors: int = 20):
        self.movies = movies.drop_duplicates("movieId").copy().reset_index(drop=True)
        self.ratings = ratings[["userId", "movieId", "rating"]].copy().reset_index(drop=True)
        self.neighbor_count = neighbors

        self.user_ids = np.sort(self.ratings["userId"].unique())
        self.movie_ids = self.movies["movieId"].to_numpy(dtype=int)
        self.user_to_pos = {int(value): i for i, value in enumerate(self.user_ids)}
        self.movie_to_pos = {int(value): i for i, value in enumerate(self.movie_ids)}

        known = self.ratings["movieId"].isin(self.movie_to_pos)
        self.ratings = self.ratings.loc[known].reset_index(drop=True)
        user_pos = self.ratings["userId"].map(self.user_to_pos).to_numpy(dtype=int)
        movie_pos = self.ratings["movieId"].map(self.movie_to_pos).to_numpy(dtype=int)
        rating_values = self.ratings["rating"].to_numpy(dtype=np.float32)

        self.rating_matrix = np.zeros((len(self.user_ids), len(self.movie_ids)), dtype=np.float32)
        self.rating_matrix[user_pos, movie_pos] = rating_values
        self.observed = self.rating_matrix > 0
        self.global_mean = float(rating_values.mean())
        counts_per_user = self.observed.sum(axis=1)
        self.user_means = np.divide(
            self.rating_matrix.sum(axis=1),
            counts_per_user,
            out=np.full(len(self.user_ids), self.global_mean, dtype=np.float32),
            where=counts_per_user > 0,
        )

        self.deviations = np.where(
            self.observed,
            self.rating_matrix - self.user_means[:, None],
            0.0,
        ).astype(np.float32)
        vector_norms = np.linalg.norm(self.deviations, axis=1)
        normalized = np.divide(
            self.deviations,
            vector_norms[:, None],
            out=np.zeros_like(self.deviations),
            where=vector_norms[:, None] > 0,
        )
        self.user_similarity = normalized @ normalized.T
        np.clip(self.user_similarity, 0.0, 1.0, out=self.user_similarity)
        np.fill_diagonal(self.user_similarity, 0.0)
        k = min(self.neighbor_count, max(1, len(self.user_ids) - 1))
        self.neighbor_positions = np.argpartition(self.user_similarity, -k, axis=1)[:, -k:]
        self.neighbor_weights = np.take_along_axis(
            self.user_similarity, self.neighbor_positions, axis=1
        )
        order = np.argsort(self.neighbor_weights, axis=1)[:, ::-1]
        self.neighbor_positions = np.take_along_axis(self.neighbor_positions, order, axis=1)
        self.neighbor_weights = np.take_along_axis(self.neighbor_weights, order, axis=1)

        rating_totals = self.rating_matrix.sum(axis=0)
        rating_counts = self.observed.sum(axis=0)
        self.movie_means = (rating_totals + 10.0 * self.global_mean) / (rating_counts + 10.0)
        self.popularity = self._popularity_scores()
        self.genre_names, self.genre_matrix = self._make_genre_matrix()
        self.genre_norms = np.linalg.norm(self.genre_matrix, axis=1)

    def _popularity_scores(self) -> np.ndarray:
        counts = self.observed.sum(axis=0).astype(float)
        averages = np.divide(
            self.rating_matrix.sum(axis=0),
            counts,
            out=np.full(len(self.movie_ids), self.global_mean),
            where=counts > 0,
        )
        prior_weight = 25.0
        return (counts * averages + prior_weight * self.global_mean) / (counts + prior_weight)

    def _make_genre_matrix(self) -> tuple[list[str], np.ndarray]:
        genres_per_movie = [
            [] if value in ("(no genres listed)", "") else str(value).split("|")
            for value in self.movies["genres"].fillna("")
        ]
        genre_names = sorted({genre for genres in genres_per_movie for genre in genres})
        genre_to_pos = {genre: i for i, genre in enumerate(genre_names)}
        matrix = np.zeros((len(self.movies), len(genre_names)), dtype=np.float32)
        for movie_pos, genres in enumerate(genres_per_movie):
            for genre in genres:
                matrix[movie_pos, genre_to_pos[genre]] = 1.0
        return genre_names, matrix

    def _user_position(self, user_id: int) -> int:
        if int(user_id) not in self.user_to_pos:
            raise ValueError(f"User {user_id} is not present in the ratings data.")
        return self.user_to_pos[int(user_id)]

    def _movie_position(self, movie_id: int) -> int:
        if int(movie_id) not in self.movie_to_pos:
            raise ValueError(f"Movie {movie_id} is not present in the movie data.")
        return self.movie_to_pos[int(movie_id)]

    def collaborative_scores(self, user_id: int) -> np.ndarray:
        """Predict each movie rating from similar users, with a mean fallback."""
        user_pos = self._user_position(user_id)
        neighbors = self.neighbor_positions[user_pos]
        weights = self.neighbor_weights[user_pos]
        neighbor_deviations = self.deviations[neighbors]
        neighbor_observed = self.observed[neighbors].astype(np.float32)
        weighted_change = weights @ neighbor_deviations
        weight_sum = weights @ neighbor_observed
        scores = self.user_means[user_pos] + np.divide(
            weighted_change,
            weight_sum,
            out=self.movie_means.astype(np.float32, copy=True) - self.user_means[user_pos],
            where=weight_sum > 1e-8,
        )
        return np.clip(scores, 0.5, 5.0)

    def content_scores(self, user_id: int) -> np.ndarray:
        """Measure genre affinity using the user's highest-rated training movies."""
        user_pos = self._user_position(user_id)
        user_ratings = self.rating_matrix[user_pos]
        liked = np.flatnonzero((user_ratings >= 4.0) & self.observed[user_pos])
        if len(liked) == 0:
            liked = np.flatnonzero(self.observed[user_pos])
            if len(liked) == 0:
                return np.zeros(len(self.movie_ids), dtype=np.float32)
            liked = liked[np.argsort(user_ratings[liked])[-min(5, len(liked)):]]

        preference_weights = np.maximum(user_ratings[liked] - 3.0, 0.5)
        profile = preference_weights @ self.genre_matrix[liked]
        profile_norm = float(np.linalg.norm(profile))
        if profile_norm == 0:
            return np.zeros(len(self.movie_ids), dtype=np.float32)
        denominator = self.genre_norms * profile_norm
        return np.divide(
            self.genre_matrix @ profile,
            denominator,
            out=np.zeros(len(self.movie_ids), dtype=np.float32),
            where=denominator > 0,
        )

    @staticmethod
    def _scale_ratings(scores: np.ndarray) -> np.ndarray:
        return np.clip((scores - 0.5) / 4.5, 0.0, 1.0)

    def recommend_for_user(self, user_id: int, top_n: int = 10,
                           method: str = "hybrid") -> pd.DataFrame:
        """Return unrated titles ordered by hybrid, collaborative, or content score."""
        user_pos = self._user_position(user_id)
        collaborative = self.collaborative_scores(user_id)
        content = self.content_scores(user_id)
        popular = self._scale_ratings(self.popularity.astype(np.float32))
        if method == "collaborative":
            combined = self._scale_ratings(collaborative)
        elif method == "content":
            combined = content
        elif method == "hybrid":
            combined = 0.60 * self._scale_ratings(collaborative) + 0.30 * content + 0.10 * popular
        else:
            raise ValueError("method must be 'hybrid', 'collaborative', or 'content'")

        unrated = ~self.observed[user_pos]
        candidates = np.flatnonzero(unrated)
        if len(candidates) == 0:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score", "predicted_rating"])
        order = candidates[np.argsort(combined[candidates])[::-1][:top_n]]
        output = self.movies.iloc[order][["movieId", "title", "genres"]].copy()
        output["score"] = combined[order]
        output["predicted_rating"] = collaborative[order]
        output["score"] = output["score"].round(3).to_numpy()
        output["predicted_rating"] = output["predicted_rating"].round(2).to_numpy()
        return output.reset_index(drop=True)

    def similar_movies(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Return genre-similar movies for an exact or partial title."""
        query = title.casefold().strip()
        exact = self.movies["title"].str.casefold() == query
        matches = np.flatnonzero(exact.to_numpy())
        if len(matches) == 0:
            partial = self.movies["title"].str.casefold().str.contains(query, regex=False)
            matches = np.flatnonzero(partial.to_numpy())
        if len(matches) == 0:
            raise ValueError(f"No movie title matched {title!r}.")
        movie_pos = int(matches[0])
        genre_vector = self.genre_matrix[movie_pos]
        denominator = self.genre_norms * max(float(np.linalg.norm(genre_vector)), 1e-8)
        scores = np.divide(
            self.genre_matrix @ genre_vector,
            denominator,
            out=np.zeros(len(self.movie_ids), dtype=np.float32),
            where=denominator > 0,
        )
        scores[movie_pos] = -1
        order = np.argsort(scores)[::-1]
        order = order[scores[order] > 0][:top_n]
        output = self.movies.iloc[order][["movieId", "title", "genres"]].copy()
        output["genre_similarity"] = scores[order].round(3)
        return output.reset_index(drop=True)

    def popular_movies(self, top_n: int = 10, minimum_ratings: int = 50) -> pd.DataFrame:
        counts = self.observed.sum(axis=0)
        eligible = np.flatnonzero(counts >= minimum_ratings)
        order = eligible[np.argsort(self.popularity[eligible])[::-1][:top_n]]
        output = self.movies.iloc[order][["movieId", "title", "genres"]].copy()
        output["weighted_rating"] = self.popularity[order].round(3)
        output["rating_count"] = counts[order]
        return output.reset_index(drop=True)


def _rating_errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    difference = predicted - actual
    return {
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "mae": float(np.mean(np.abs(difference))),
    }


def evaluate_system(movies: pd.DataFrame, ratings: pd.DataFrame,
                    test_fraction: float = 0.20, seed: int = RANDOM_SEED,
                    top_n: int = 10) -> dict:
    """Evaluate held-out rating error and top-N retrieval of held-out likes."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("userId", sort=True):
        order = rng.permutation(len(group))
        test_count = max(1, int(round(len(group) * test_fraction)))
        test_parts.append(group.iloc[order[:test_count]])
        train_parts.append(group.iloc[order[test_count:]])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    model = MovieRecommendationSystem(movies, train)
    actual, cf_predictions = [], []
    precision_values, recall_values, hit_values = [], [], []
    liked_threshold = 4.0

    for user_id, held_out in test.groupby("userId", sort=True):
        user_pos = model._user_position(int(user_id))
        scores = model.collaborative_scores(int(user_id))
        held_movie_positions = held_out["movieId"].map(model.movie_to_pos).to_numpy(dtype=int)
        cf_predictions.extend(scores[held_movie_positions].astype(float).tolist())
        actual.extend(held_out["rating"].astype(float).tolist())

        liked = set(held_out.loc[held_out["rating"] >= liked_threshold, "movieId"].astype(int))
        if liked:
            recommendations = model.recommend_for_user(int(user_id), top_n=top_n, method="hybrid")
            recommended_ids = recommendations["movieId"].astype(int).tolist()
            hits = len(liked.intersection(recommended_ids))
            precision_values.append(hits / top_n)
            recall_values.append(hits / len(liked))
            hit_values.append(1.0 if hits else 0.0)

    actual_array = np.asarray(actual, dtype=float)
    cf_array = np.asarray(cf_predictions, dtype=float)
    global_array = np.full_like(actual_array, float(train["rating"].mean()))
    return {
        "split": {
            "strategy": "Per-user random 80/20 rating holdout",
            "random_seed": seed,
            "train_ratings": int(len(train)),
            "test_ratings": int(len(test)),
            "test_fraction": test_fraction,
        },
        "rating_prediction": {
            "global_mean_baseline": _rating_errors(actual_array, global_array),
            "user_user_collaborative_filtering": _rating_errors(actual_array, cf_array),
        },
        "hybrid_ranking": {
            "positive_rating_threshold": liked_threshold,
            "top_n": top_n,
            "users_with_held_out_likes": len(recall_values),
            "precision_at_10": float(np.mean(precision_values)) if precision_values else 0.0,
            "recall_at_10": float(np.mean(recall_values)) if recall_values else 0.0,
            "hit_rate_at_10": float(np.mean(hit_values)) if hit_values else 0.0,
        },
    }


def run_project(evaluate: bool = True, user_id: int = 1) -> dict:
    movies, ratings = load_dataset()
    model = MovieRecommendationSystem(movies, ratings)
    results = {
        "project_title": "Movie Recommendation System Using Hybrid Filtering",
        "dataset": {
            "users": int(ratings["userId"].nunique()),
            "movies": int(movies["movieId"].nunique()),
            "ratings": int(len(ratings)),
            "rating_scale": "0.5 to 5.0 stars",
        },
        "sample_user_id": int(user_id),
        "sample_hybrid_recommendations": model.recommend_for_user(user_id, top_n=5).to_dict(orient="records"),
        "sample_content_recommendations": model.similar_movies("Toy Story (1995)", top_n=5).to_dict(orient="records"),
    }
    if evaluate:
        results["evaluation"] = evaluate_system(movies, ratings)

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def print_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("No recommendations available for this input.")
        return
    print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens content-based and hybrid movie recommender")
    commands = parser.add_subparsers(dest="command", required=True)

    user_parser = commands.add_parser("user", help="Recommend movies for a MovieLens user ID")
    user_parser.add_argument("--user-id", type=int, default=1)
    user_parser.add_argument("--method", choices=["hybrid", "collaborative", "content"], default="hybrid")
    user_parser.add_argument("--top", type=int, default=10)

    similar_parser = commands.add_parser("similar", help="Find movies with similar genres")
    similar_parser.add_argument("--title", default="Toy Story (1995)")
    similar_parser.add_argument("--top", type=int, default=10)

    popular_parser = commands.add_parser("popular", help="Show popular movies with a rating-count threshold")
    popular_parser.add_argument("--top", type=int, default=10)

    evaluate_parser = commands.add_parser("evaluate", help="Run the reproducible holdout evaluation")
    evaluate_parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    commands.add_parser("summary", help="Save the complete recommendations and evaluation summary to outputs/")

    args = parser.parse_args()
    if args.command == "summary":
        print(json.dumps(run_project(), indent=2))
        return
    movies, ratings = load_dataset()
    if args.command == "evaluate":
        results = evaluate_system(movies, ratings, seed=args.seed)
        print(json.dumps(results, indent=2))
        return

    recommender = MovieRecommendationSystem(movies, ratings)
    if args.command == "user":
        print(f"Top {args.top} {args.method} recommendations for anonymized MovieLens user {args.user_id}:")
        print_table(recommender.recommend_for_user(args.user_id, top_n=args.top, method=args.method))
    elif args.command == "similar":
        print(f"Movies similar by genre to {args.title!r}:")
        print_table(recommender.similar_movies(args.title, top_n=args.top))
    elif args.command == "popular":
        print_table(recommender.popular_movies(top_n=args.top))


if __name__ == "__main__":
    main()
