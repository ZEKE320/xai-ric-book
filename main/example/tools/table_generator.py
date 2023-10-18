import pandas as pd
from tabulate import tabulate


class TableCreator:
    def display(self, df: pd.DataFrame):
        print(tabulate(df, headers="keys", tablefmt="psql"))
