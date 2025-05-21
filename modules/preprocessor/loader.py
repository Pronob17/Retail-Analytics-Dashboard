import os

import pandas as pd

class LoaderClass:
    def __init__(self, demo_df=None):
        self.demo_df = demo_df
        self.main_file = None

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
        self.main_file = main_file
        main_df = pd.read_csv(self.main_file)

        return main_df
