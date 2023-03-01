from pathlib import Path
import pandas as pd

from config import root_path, out_path, Fields


def read_census_file(root_path, year):
    F = Fields()
    census_df = pd.read_csv(Path(root_path) / f"ACSST5Y{year}.S0101-Data.csv", usecols=['GEO_ID', 'S0101_C01_026E'])
    census_df = census_df.iloc[1:]
    census_df["S0101_C01_026E"] = census_df["S0101_C01_026E"].astype(int)
    census_df["GEO_ID"] = census_df["GEO_ID"].astype(str).str.split("US").str.get(-1).astype(int)
    census_df.rename(columns={"GEO_ID": F.cfips, "S0101_C01_026E": F.pop}, inplace=True)
    census_df["year"] = year + 2
    return census_df


if __name__ == "__main__":
    census_dfs = []
    for year in [2017, 2018, 2019, 2020, 2021]:
        census_df = read_census_file(root_path, year)
        census_dfs.append(census_df)
    census_dfs = pd.concat(census_dfs)
    print(census_dfs.shape)
    census_dfs.to_csv(Path(out_path) / "populations.csv", index=False)
