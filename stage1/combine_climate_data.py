from __future__ import annotations

from pathlib import Path

import pandas as pd


MEASUREMENT_COLUMNS = [
	"temperature",
	"dew_point_temperature",
	"station_level_pressure",
	"wind_speed",
	"precipitation",
	"relative_humidity",
	"visibility",
	"snow_depth",
]


def load_year_data(data_dir: Path, year: int, stations: list[str]) -> pd.DataFrame:
	frames: list[pd.DataFrame] = []
	for station in stations:
		file_path = data_dir / f"{station}_{year}.psv"
		df = pd.read_csv(file_path, sep="|", dtype=str)
		frames.append(df)

	combined = pd.concat(frames, ignore_index=True)
	key_columns = ["Year", "Month", "Day", "Hour"]
	available_measurements = [
		column for column in MEASUREMENT_COLUMNS if column in combined.columns
	]
	keep_columns = key_columns + available_measurements
	combined = combined[keep_columns]

	for column in available_measurements:
		combined[column] = pd.to_numeric(combined[column], errors="coerce")

	combined["Year"] = pd.to_numeric(combined["Year"], errors="coerce")
	combined["Month"] = pd.to_numeric(combined["Month"], errors="coerce")
	combined["Day"] = pd.to_numeric(combined["Day"], errors="coerce")
	combined["Hour"] = pd.to_numeric(combined["Hour"], errors="coerce")

	grouped = combined.groupby(key_columns, dropna=False)
	mean_df = grouped[available_measurements].mean().add_suffix("_mean")
	std_df = grouped[available_measurements].std(ddof=1).add_suffix("_std")

	result = pd.concat([mean_df, std_df], axis=1).reset_index()
	return result


def main() -> None:
	project_root = Path(__file__).resolve().parent
	data_dir = project_root / "dataset" / "climate"
	stations = ["burbank", "downtown", "elmonte", "whiteman"]

	results: list[pd.DataFrame] = []
	for year in (2018, 2019):
		result = load_year_data(data_dir, year, stations)
		results.append(result)

	combined = pd.concat(results, ignore_index=True).sort_values(
		["Year", "Month", "Day", "Hour"]
	)
	output_path = data_dir / "average_all.psv"
	combined.to_csv(output_path, sep="|", index=False)


if __name__ == "__main__":
	main()
