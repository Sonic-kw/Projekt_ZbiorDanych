import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils.logger import setup_logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = setup_logger("statistics")

class MarketAnalysis:
    POLISH_COLUMN_LABELS = {
        "brand": "Marka",
        "model": "Model",
        "year": "Rok produkcji",
        "mileage": "Przebieg",
        "price": "Cena",
        "capacity": "Pojemność silnika",
        "power": "Moc",
        "type": "Typ motocykla",
        "cluster": "Klaster",
        "brand_cluster": "Klaster marek",
        "n": "Liczba ogłoszeń",
        "avg_price": "Średnia cena",
        "avg_mileage": "Średni przebieg",
        "price_year_slope": "Współczynnik utraty wartości",
        "price_year_r2": "R2 ceny",
        "mileage_year_slope": "Nachylenie przebiegu",
        "mileage_year_r2": "R2 przebiegu",
        "usage_km_per_year": "Współczynnik eksploatacji",
        "benchmark_count": "Liczba podobnych ogłoszeń",
        "benchmark_price": "Średnia cena podobnych",
        "benchmark_mileage": "Średni przebieg podobnych",
        "price_delta_pct": "Cena względem średniej (%)",
        "mileage_delta_pct": "Przebieg względem średniej (%)",
        "deal_score": "Wynik okazji",
        "missing_pct": "Braki danych (%)",
    }

    POLISH_FILENAME_PARTS = {
        "brand": "marka",
        "model": "model",
        "year": "rok_produkcji",
        "mileage": "przebieg",
        "price": "cena",
        "capacity": "pojemnosc",
        "power": "moc",
        "type": "typ",
    }

    @staticmethod
    def _ensure_output_dir(output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _label(cls, column: str) -> str:
        return cls.POLISH_COLUMN_LABELS.get(column, column)

    @classmethod
    def _filename_part(cls, column: str) -> str:
        return cls.POLISH_FILENAME_PARTS.get(column, column)

    @classmethod
    def localize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy with Polish column and index labels for exported CSV files."""
        return df.rename(columns=cls.POLISH_COLUMN_LABELS, index=cls.POLISH_COLUMN_LABELS)

    @staticmethod
    def _present_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
        return [column for column in columns if column in df.columns]

    @staticmethod
    def _top_values(df: pd.DataFrame, column: str, top_n: int) -> list:
        if column not in df.columns:
            return []
        return df[column].value_counts().head(top_n).index.tolist()

    @staticmethod
    def calculate_market_shares(df: pd.DataFrame, column: str = "brand") -> pd.Series:
        """Calculates the percentage share of each brand."""
        if column not in df.columns:
            logger.warning(f"Column {column} missing. Cannot calculate market shares.")
            return pd.Series(dtype=float)
        shares = df[column].value_counts(normalize=True) * 100
        return shares

    @staticmethod
    def calculate_percentile_summary(
        df: pd.DataFrame,
        columns: list[str] | None = None,
        percentiles: tuple[float, ...] = (0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1),
    ) -> pd.DataFrame:
        """Summarizes numeric ranges before/after cleaning, inspired by the reference analysis."""
        columns = columns or ["price", "mileage", "year", "capacity", "power"]
        numeric_columns = MarketAnalysis._present_columns(df, columns)
        if not numeric_columns:
            return pd.DataFrame()

        summary = df[numeric_columns].quantile(percentiles).T
        summary.columns = [f"p{int(percentile * 100):02d}" for percentile in percentiles]
        summary["missing_pct"] = df[numeric_columns].isna().mean() * 100
        return summary.round(2)

    @staticmethod
    def calculate_brand_regression_coefficients(
        df: pd.DataFrame,
        min_records: int = 8,
    ) -> pd.DataFrame:
        """
        Calculates per-brand slopes for price and mileage vs production year.

        Higher price_year_slope means the brand loses more PLN with every year of age.
        More negative mileage_year_slope means older listings tend to have much higher mileage.
        """
        required = {"brand", "year", "price", "mileage"}
        if not required.issubset(df.columns):
            missing = ", ".join(sorted(required - set(df.columns)))
            logger.warning(f"Missing columns for brand regressions: {missing}")
            return pd.DataFrame()

        rows = []
        for brand, brand_df in df.dropna(subset=list(required)).groupby("brand"):
            if len(brand_df) < min_records or brand_df["year"].nunique() < 2:
                continue

            X = brand_df[["year"]]
            price_model = LinearRegression().fit(X, brand_df["price"])
            mileage_model = LinearRegression().fit(X, brand_df["mileage"])
            price_pred = price_model.predict(X)
            mileage_pred = mileage_model.predict(X)

            rows.append({
                "brand": brand,
                "n": len(brand_df),
                "avg_price": brand_df["price"].mean(),
                "avg_mileage": brand_df["mileage"].mean(),
                "price_year_slope": price_model.coef_[0],
                "price_year_r2": r2_score(brand_df["price"], price_pred),
                "mileage_year_slope": mileage_model.coef_[0],
                "mileage_year_r2": r2_score(brand_df["mileage"], mileage_pred),
                "usage_km_per_year": -mileage_model.coef_[0],
            })

        return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    @staticmethod
    def plot_market_structure(
        df: pd.DataFrame,
        output_path: str = "results/struktura_rynku.png",
        top_n: int = 15,
    ):
        """Plots brand shares and the strongest secondary categorical split."""
        MarketAnalysis._ensure_output_dir(output_path)
        if df.empty or "brand" not in df.columns:
            logger.warning("No brand data available. Skipping market structure plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        brand_counts = df["brand"].value_counts().head(top_n).sort_values()
        brand_share = (brand_counts / len(df) * 100).round(1)
        sns.barplot(x=brand_share.values, y=brand_share.index, ax=axes[0], hue=brand_share.index, palette="viridis", legend=False)
        axes[0].set_title(f"Top {top_n} marek według udziału w rynku")
        axes[0].set_xlabel("Udział w ogłoszeniach (%)")
        axes[0].set_ylabel("Marka")
        axes[0].grid(axis="x", linestyle="--", alpha=0.5)

        split_column = "type" if "type" in df.columns and df["type"].nunique() > 1 else "model"
        split_counts = df[split_column].value_counts().head(top_n).sort_values()
        split_share = (split_counts / len(df) * 100).round(1)
        sns.barplot(x=split_share.values, y=split_share.index, ax=axes[1], hue=split_share.index, palette="mako", legend=False)
        axes[1].set_title(f"Top {top_n}: {MarketAnalysis._label(split_column)}")
        axes[1].set_xlabel("Udział w ogłoszeniach (%)")
        axes[1].set_ylabel(MarketAnalysis._label(split_column))
        axes[1].grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Market structure chart saved to {output_path}")

    @staticmethod
    def plot_numeric_diagnostics(
        df: pd.DataFrame,
        output_path: str = "results/diagnostyka_percentyli.png",
    ):
        """Plots percentile diagnostics for numeric fields used during cleaning."""
        MarketAnalysis._ensure_output_dir(output_path)
        summary = MarketAnalysis.calculate_percentile_summary(df)
        if summary.empty:
            logger.warning("No numeric columns available. Skipping numeric diagnostics plot.")
            return

        heatmap_data = summary.drop(columns=["missing_pct"], errors="ignore")
        heatmap_data = heatmap_data.rename(index=MarketAnalysis.POLISH_COLUMN_LABELS)
        plt.figure(figsize=(14, max(4, 0.8 * len(heatmap_data))))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
        plt.title("Percentyle cech numerycznych używane w diagnostyce czyszczenia")
        plt.xlabel("Percentyl")
        plt.ylabel("Cecha")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Numeric percentile diagnostics chart saved to {output_path}")

    @staticmethod
    def plot_distributions(df: pd.DataFrame, output_path: str = "results/rozklady_ceny_przebiegu.png"):
        """Plots overall price and mileage distributions with mean and median markers."""
        MarketAnalysis._ensure_output_dir(output_path)
        required_columns = MarketAnalysis._present_columns(df, ["price", "mileage"])
        if len(required_columns) < 2:
            logger.warning("Price or mileage missing. Skipping distributions plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, column, unit, color in [
            (axes[0], "price", "PLN", "steelblue"),
            (axes[1], "mileage", "km", "seagreen"),
        ]:
            sns.histplot(df[column], kde=True, ax=ax, color=color)
            ax.axvline(df[column].mean(), color="red", linewidth=2, label="Średnia")
            ax.axvline(df[column].median(), color="navy", linewidth=2, label="Mediana")
            ax.set_title(f"Rozkład: {MarketAnalysis._label(column)}")
            ax.set_xlabel(f"{MarketAnalysis._label(column)} ({unit})")
            ax.set_ylabel("Liczba ogłoszeń")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Price and mileage distributions chart saved to {output_path}")

    @staticmethod
    def plot_brand_distributions(
        df: pd.DataFrame,
        output_path: str = "results/rozklady_wedlug_marek.png",
        top_n: int = 12,
    ):
        """Plots price and mileage spread for the most common brands."""
        MarketAnalysis._ensure_output_dir(output_path)
        required = {"brand", "price", "mileage"}
        if not required.issubset(df.columns):
            logger.warning("Brand, price, or mileage missing. Skipping brand distributions plot.")
            return

        top_brands = MarketAnalysis._top_values(df, "brand", top_n)
        plot_df = df[df["brand"].isin(top_brands)].copy()
        if plot_df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        order = plot_df.groupby("brand")["price"].median().sort_values().index
        sns.boxplot(data=plot_df, x="price", y="brand", order=order, ax=axes[0], hue="brand", palette="viridis", legend=False)
        axes[0].set_title("Zróżnicowanie cen według marek")
        axes[0].set_xlabel("Cena (PLN)")
        axes[0].set_ylabel("Marka")
        axes[0].grid(axis="x", linestyle="--", alpha=0.4)

        order = plot_df.groupby("brand")["mileage"].median().sort_values().index
        sns.boxplot(data=plot_df, x="mileage", y="brand", order=order, ax=axes[1], hue="brand", palette="mako", legend=False)
        axes[1].set_title("Zróżnicowanie przebiegu według marek")
        axes[1].set_xlabel("Przebieg (km)")
        axes[1].set_ylabel("Marka")
        axes[1].grid(axis="x", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Brand distributions chart saved to {output_path}")

    @staticmethod
    def plot_regression_analysis(
        df: pd.DataFrame,
        x_col: str,
        y_col: str = "price",
        output_path: str | None = None,
    ):
        """Plots a scatter plot with a linear regression line."""
        output_path = output_path or (
            f"results/regresja_{MarketAnalysis._filename_part(y_col)}_od_"
            f"{MarketAnalysis._filename_part(x_col)}.png"
        )
        MarketAnalysis._ensure_output_dir(output_path)
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            logger.warning(f"Columns {x_col} or {y_col} missing. Skipping regression plot.")
            return

        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "red"})
        plt.title(f"Zależność: {MarketAnalysis._label(y_col)} od {MarketAnalysis._label(x_col)}", fontsize=15)
        plt.xlabel(MarketAnalysis._label(x_col))
        plt.ylabel(MarketAnalysis._label(y_col))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Regression chart saved to {output_path}")

    @staticmethod
    def plot_relationship_facets(
        df: pd.DataFrame,
        x_col: str,
        y_col: str = "price",
        output_path: str | None = None,
        top_n: int = 12,
    ):
        """Facets a relationship by brand, similar in spirit to the reference article."""
        output_path = output_path or (
            f"results/{MarketAnalysis._filename_part(y_col)}_od_"
            f"{MarketAnalysis._filename_part(x_col)}_wedlug_marek.png"
        )
        MarketAnalysis._ensure_output_dir(output_path)
        required = {"brand", x_col, y_col}
        if not required.issubset(df.columns):
            logger.warning(f"Missing columns for faceted relationship plot: {required - set(df.columns)}")
            return

        top_brands = MarketAnalysis._top_values(df, "brand", top_n)
        plot_df = df[df["brand"].isin(top_brands)].copy()
        if plot_df.empty:
            return

        grid = sns.FacetGrid(plot_df, col="brand", col_wrap=4, height=3.1, sharex=False, sharey=False)
        grid.map_dataframe(
            sns.regplot,
            x=x_col,
            y=y_col,
            scatter_kws={"alpha": 0.25, "s": 12},
            line_kws={"color": "red", "linewidth": 1.5},
        )
        grid.set_titles("{col_name}")
        grid.set_axis_labels(MarketAnalysis._label(x_col), MarketAnalysis._label(y_col))
        grid.figure.suptitle(
            f"{MarketAnalysis._label(y_col)} od {MarketAnalysis._label(x_col)} według marek",
            y=1.02,
        )
        grid.figure.tight_layout()
        grid.figure.savefig(output_path)
        plt.close(grid.figure)
        logger.info(f"Faceted relationship chart saved to {output_path}")

    @staticmethod
    def plot_year_brand_heatmap(
        df: pd.DataFrame,
        output_path: str = "results/mapa_marek_rocznikow.png",
        top_n: int = 20,
    ):
        """Plots listing concentration by brand and production year."""
        MarketAnalysis._ensure_output_dir(output_path)
        required = {"brand", "year"}
        if not required.issubset(df.columns):
            logger.warning("Brand or year missing. Skipping year-brand heatmap.")
            return

        top_brands = MarketAnalysis._top_values(df, "brand", top_n)
        plot_df = df[df["brand"].isin(top_brands)]
        pivot = plot_df.pivot_table(index="brand", columns="year", values="price", aggfunc="count", fill_value=0)
        if pivot.empty:
            return

        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        plt.figure(figsize=(16, max(6, 0.4 * len(pivot))))
        sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.2, linecolor="white")
        plt.title("Liczba ogłoszeń według marki i roku produkcji")
        plt.xlabel("Rok produkcji")
        plt.ylabel("Marka")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Brand-year heatmap saved to {output_path}")

    @staticmethod
    def plot_model_year_heatmap(
        df: pd.DataFrame,
        output_path: str = "results/mapa_modeli_rocznikow.png",
        top_brands: int = 8,
        models_per_brand: int = 4,
    ):
        """Plots the most common models within top brands across production years."""
        MarketAnalysis._ensure_output_dir(output_path)
        required = {"brand", "model", "year"}
        if not required.issubset(df.columns):
            logger.warning("Brand, model, or year missing. Skipping model-year heatmap.")
            return

        selected_brands = MarketAnalysis._top_values(df, "brand", top_brands)
        selected_frames = []
        for brand in selected_brands:
            brand_df = df[df["brand"] == brand]
            selected_models = MarketAnalysis._top_values(brand_df, "model", models_per_brand)
            selected_frames.append(brand_df[brand_df["model"].isin(selected_models)])

        if not selected_frames:
            return

        plot_df = pd.concat(selected_frames).copy()
        plot_df["brand_model"] = plot_df["brand"].astype(str) + " / " + plot_df["model"].astype(str)
        pivot = plot_df.pivot_table(index="brand_model", columns="year", values="price", aggfunc="count", fill_value=0)
        if pivot.empty:
            return

        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        plt.figure(figsize=(17, max(7, 0.35 * len(pivot))))
        sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.2, linecolor="white")
        plt.title("Liczba ogłoszeń według roku produkcji dla popularnych marek i modeli")
        plt.xlabel("Rok produkcji")
        plt.ylabel("Marka / model")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Model-year heatmap saved to {output_path}")

    @staticmethod
    def plot_depreciation_exploitation(
        df: pd.DataFrame,
        output_path: str = "results/utrata_wartosci_i_eksploatacja.png",
        min_records: int = 8,
        n_clusters: int = 4,
    ) -> pd.DataFrame:
        """Plots brand-level depreciation and exploitation coefficients."""
        MarketAnalysis._ensure_output_dir(output_path)
        coefficients = MarketAnalysis.calculate_brand_regression_coefficients(df, min_records=min_records)
        if coefficients.empty:
            logger.warning("No brand coefficients available. Skipping depreciation/exploitation plot.")
            return coefficients

        cluster_columns = ["price_year_slope", "usage_km_per_year"]
        if len(coefficients) >= n_clusters:
            scaled = StandardScaler().fit_transform(coefficients[cluster_columns])
            coefficients["brand_cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(scaled)
        else:
            coefficients["brand_cluster"] = 0

        fig, axes = plt.subplots(1, 3, figsize=(22, max(7, 0.35 * len(coefficients))))

        price_rank = coefficients.sort_values("price_year_slope")
        sns.scatterplot(data=price_rank, x="price_year_slope", y="brand", ax=axes[0], color="red", s=80)
        axes[0].set_title("Współczynnik utraty wartości")
        axes[0].set_xlabel("Nachylenie ceny względem roku (PLN/rok)")
        axes[0].set_ylabel("Marka")
        axes[0].grid(axis="x", linestyle="--", alpha=0.4)

        mileage_rank = coefficients.sort_values("usage_km_per_year")
        sns.scatterplot(data=mileage_rank, x="usage_km_per_year", y="brand", ax=axes[1], color="darkgreen", s=80)
        axes[1].set_title("Współczynnik eksploatacji")
        axes[1].set_xlabel("Szacowany przebieg roczny (km/rok)")
        axes[1].set_ylabel("")
        axes[1].grid(axis="x", linestyle="--", alpha=0.4)

        sns.scatterplot(
            data=coefficients,
            x="price_year_slope",
            y="usage_km_per_year",
            hue="brand_cluster",
            palette="deep",
            s=100,
            ax=axes[2],
        )
        for _, row in coefficients.iterrows():
            axes[2].text(row["price_year_slope"], row["usage_km_per_year"], row["brand"], fontsize=8, alpha=0.8)
        axes[2].set_title("Klastry marek według utraty wartości i eksploatacji")
        axes[2].set_xlabel("Współczynnik utraty wartości (PLN/rok)")
        axes[2].set_ylabel("Współczynnik eksploatacji (km/rok)")
        axes[2].grid(True, linestyle="--", alpha=0.4)
        axes[2].legend(title="Klaster")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Depreciation and exploitation chart saved to {output_path}")
        return coefficients

    @staticmethod
    def plot_bargain_candidates(
        bargains: pd.DataFrame,
        output_path: str = "results/okazje_rynkowe.png",
    ):
        """Plots candidate bargains against peer price and mileage averages."""
        MarketAnalysis._ensure_output_dir(output_path)
        required = {"price_delta_pct", "mileage_delta_pct", "brand", "model"}
        fig, ax = plt.subplots(figsize=(11, 8))

        if bargains.empty or not required.issubset(bargains.columns):
            ax.text(0.5, 0.5, "Nie znaleziono kandydatów na okazje", ha="center", va="center", fontsize=14)
            ax.set_axis_off()
        else:
            plot_df = bargains.head(30).copy()
            sns.scatterplot(
                data=plot_df,
                x="price_delta_pct",
                y="mileage_delta_pct",
                hue="brand",
                size="deal_score" if "deal_score" in plot_df.columns else None,
                sizes=(60, 260),
                alpha=0.8,
                ax=ax,
            )
            for _, row in plot_df.iterrows():
                label = f"{row['brand']} {row['model']}"
                ax.text(row["price_delta_pct"], row["mileage_delta_pct"], label, fontsize=8, alpha=0.75)
            ax.axvline(0, color="gray", linewidth=1)
            ax.axhline(0, color="gray", linewidth=1)
            ax.set_title("Kandydaci na okazje względem średnich dla podobnych ogłoszeń")
            ax.set_xlabel("Cena względem średniej (%)")
            ax.set_ylabel("Przebieg względem średniej (%)")
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Bargain candidates chart saved to {output_path}")

    @staticmethod
    def plot_clusters(df: pd.DataFrame, output_path: str = "results/klastry_ogloszen.png"):
        """Plots K-means clusters (Price vs Mileage)."""
        MarketAnalysis._ensure_output_dir(output_path)
        if 'cluster' not in df.columns:
            logger.error("DataFrame does not contain 'cluster' column. Skipping cluster plot.")
            return

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='mileage', y='price', hue='cluster', palette='deep', s=100, alpha=0.7)
        plt.title("Segmentacja ogłoszeń (klasteryzacja k-means)", fontsize=15)
        plt.xlabel("Przebieg (km)")
        plt.ylabel("Cena (PLN)")
        plt.legend(title="Klaster")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Listing clusters chart saved to {output_path}")

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, output_path: str = "results/korelacje_cech.png"):
        """Plots a heatmap of the correlation matrix."""
        MarketAnalysis._ensure_output_dir(output_path)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            logger.warning("No numeric columns available. Skipping correlation matrix.")
            return
        corr = numeric_df.corr()
        corr = corr.rename(index=MarketAnalysis.POLISH_COLUMN_LABELS, columns=MarketAnalysis.POLISH_COLUMN_LABELS)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Macierz korelacji cech motocykli", fontsize=15)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Correlation matrix chart saved to {output_path}")
