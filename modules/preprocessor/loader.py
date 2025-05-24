import os
import json

import pandas as pd

from tests.errorlog import log_error


class LoaderClass:
    def __init__(self):
        # initialize objects
        self.df = None
        self.validity = None
        self.alternative_names = None
        self.standard_columns = None

    def demo_func(self):
        """
        Merges the demo files and converts them to dataframe.
        :return: demo_df
        """
        # get the demo data from the folder
        path = "./data/"
        cust_file = os.path.join(path, "customer_data.csv")
        prod_file = os.path.join(path, "product_master.csv")
        tran_file = os.path.join(path, "transaction_data.csv")

        # create the merged file
        cust_df = pd.read_csv(cust_file)  # create df
        prod_df = pd.read_csv(prod_file)  # create df
        tran_df = pd.read_csv(tran_file)  # create df
        # now merge the file
        prod_tran_df = pd.merge(prod_df, tran_df, on="ProductID")
        demo_df = pd.merge(cust_df, prod_tran_df, on="CustomerID")

        return demo_df

    def loader_func(self, main_file):
        """
        Creates dataframe from the uploaded file.
        :param main_file:
        :return:
        """

        main_df = pd.read_csv(main_file)

        return main_df

    def standardize_func(self, df, data_path):
        """
        Standardizes the columns.
        :return: df
        """
        json_path = 'columns_dictionary_full.json'
        json_file = data_path + json_path

        # load mapping from json file
        with open(json_file, 'r') as file:
            self.alternative_names = json.load(file)

        # get standard column names from json keys
        self.standard_columns = list(self.alternative_names.keys())
        self.df = df
        matched_columns = {}
        invalid_columns = []

        # Lowercase mapping of df columns for case-insensitive matching
        df_cols_lower = {col.lower(): col for col in self.df.columns}

        for std_col in self.standard_columns:
            alternatives = self.alternative_names.get(std_col, [])
            # Prepare lowercase list of alternatives + standard name itself
            alternatives_lower = [alt.lower() for alt in alternatives] + [std_col.lower()]

            # Find first matching lowercase column name in df
            matched_col_lower = next((col for col in alternatives_lower if col in df_cols_lower), None)

            if matched_col_lower:
                # Original column name with original case from df
                original_col = df_cols_lower[matched_col_lower]
                matched_columns[std_col] = original_col
            else:
                invalid_columns.append(std_col)

        # Rename matched columns in df: original column -> standard column (standard case preserved)
        for std_col, matched_col in matched_columns.items():
            self.df.rename(columns={matched_col: std_col}, inplace=True)

        # create the necessary columns
        # create final amount if not present from selling price and quantity
        if "FinalAmount" not in self.df.columns:
            if all(col in self.df.columns for col in ["SellingPrice", "Quantity", "DiscountPercent"]):
                # Calculate FinalAmount = (SellingPrice × Quantity) × (1 - DiscountPercent / 100)
                self.df["FinalAmount"] = self.df["SellingPrice"] * self.df["Quantity"] * (
                        1 - self.df["DiscountPercent"] / 100)

        # create discount percentage from cost price, quantity and final amount
        if "DiscountPercent" not in self.df.columns:
            if all(col in self.df.columns for col in ["CostPrice", "Quantity", "FinalAmount"]):
                self.df['DiscountPercent'] = ((self.df['CostPrice'] * self.df['Quantity']) - self.df['FinalAmount']) / (
                        self.df['CostPrice'] * self.df['Quantity']) * 100

        print("📋 Columns After Renaming:", self.df.columns.tolist())

        # Create a DataFrame of matched columns
        validity_df = pd.DataFrame(
            [(std_col, matched_columns[std_col]) for std_col in matched_columns],
            columns=["Standard Column", "Matched Column"]
        )

        print("🔑 Final Validity Mapping DataFrame:")
        print(validity_df)

        return self.df, validity_df