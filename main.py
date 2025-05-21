import pandas as pd

from modules.interface.dashboard import DashboardClass
from modules.preprocessor import loader
from modules.preprocessor.cleaner import CleanerClass
from modules.preprocessor.loader import LoaderClass




def main():
    main_df = None

    # call title display
    title = DashboardClass()
    title.title_func()

    # call the loader module to get demo dataframe
    demo = LoaderClass()
    demo_df = demo.demo_func()

    # call upload display and load file
    uploaded = DashboardClass()
    selection, uploaded_file = uploaded.upload_func()

    if selection=='Upload Data' and uploaded_file:
        upload = LoaderClass()
        main_df = upload.loader_func(uploaded_file)

    elif selection=='Demo Data' and uploaded_file is None:
        demo = LoaderClass()
        main_df = demo.demo_func()

    # clean datetime of the dataframe
    clean_date = CleanerClass(main_df)
    cleaned_date_main_df = clean_date.datetime_func()

    # impute the dataframe
    clean_impute = CleanerClass(cleaned_date_main_df)
    cleaned_date_imputed_main_df = clean_impute.imputer_func()

    # call the kpi
    kpi = DashboardClass(cleaned_date_imputed_main_df)
    kpi.show_kpi_func()



if __name__ == "__main__":
    main()