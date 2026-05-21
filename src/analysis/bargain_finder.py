import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("bargain_finder")

class BargainFinder:
    def __init__(self, threshold: float = 0.15, min_group_size: int = 3):
        self.threshold = threshold
        self.min_group_size = min_group_size

    @staticmethod
    def add_peer_benchmarks(
        df: pd.DataFrame,
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Adds leave-one-out peer averages for comparable motorcycles.

        A listing is compared with other listings of the same brand, model and year.
        The current row is excluded from its benchmark so it cannot make itself look
        cheaper or more expensive.
        """
        group_columns = group_columns or ["brand", "model", "year"]
        required = set(group_columns + ["price", "mileage"])
        if not required.issubset(df.columns):
            missing = ", ".join(sorted(required - set(df.columns)))
            raise ValueError(f"Cannot calculate peer benchmarks. Missing columns: {missing}")

        group_stats = (
            df.groupby(group_columns)
            .agg(
                group_price_sum=("price", "sum"),
                group_mileage_sum=("mileage", "sum"),
                group_count=("price", "size"),
            )
            .reset_index()
        )

        result = df.merge(group_stats, on=group_columns, how="left")
        result["benchmark_count"] = result["group_count"] - 1
        result["benchmark_price"] = (result["group_price_sum"] - result["price"]) / result["benchmark_count"]
        result["benchmark_mileage"] = (result["group_mileage_sum"] - result["mileage"]) / result["benchmark_count"]
        result = result.replace([float("inf"), -float("inf")], pd.NA)

        result["price_delta_pct"] = 100 * (result["price"] / result["benchmark_price"] - 1)
        result["mileage_delta_pct"] = 100 * (result["mileage"] / result["benchmark_mileage"] - 1)
        result["deal_score"] = -result["price_delta_pct"] + -result["mileage_delta_pct"]
        return result.drop(columns=["group_price_sum", "group_mileage_sum", "group_count"])

    def find_bargains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies listings cheaper and less used than peers from the same model year.
        """
        df_with_benchmarks = self.add_peer_benchmarks(df)
        threshold_pct = -100 * self.threshold

        bargains = df_with_benchmarks[
            (df_with_benchmarks["benchmark_count"] >= self.min_group_size - 1)
            & (df_with_benchmarks["benchmark_price"] > 0)
            & (df_with_benchmarks["benchmark_mileage"] > 0)
            & (df_with_benchmarks["price_delta_pct"] <= threshold_pct)
            & (df_with_benchmarks["mileage_delta_pct"] <= threshold_pct)
        ].copy()

        bargains = bargains.sort_values(["deal_score", "benchmark_count"], ascending=[False, False])
        logger.info(f"Found {len(bargains)} potential bargains.")
        return bargains
