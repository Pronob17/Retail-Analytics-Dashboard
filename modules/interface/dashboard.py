import streamlit as st
# wide layout display
st.set_page_config(layout="wide")




class DashboardClass:
    def __init__(self, main_df=None):
        self.main_df = main_df

    def title_func(self):
        """
        Creates title display.
        :return:
        """
        # create title display
        path = "./assets/"
        lt, ct, rt = st.columns([1,3,1])
        lt.image(path+"business_image.png")
        ct.title("RETAIL ANALYTICS DASHBOARD")
        rt.image(path+"retail.png")

    def upload_func(self):
        """
        Creates upload and demo selectbox.
        :return:
        """
        options = ['Demo Data', 'Upload Data']
        selection = st.sidebar.selectbox(label="SELECT THE DATA", options=options)
        # if selection is upload data
        if selection=='Upload Data':
            upload_file = st.sidebar.file_uploader("Select file")
            # check if uploaded file exist
            if upload_file:
                st.sidebar.success("File Uploaded Successfully!!")
                return selection, upload_file
            # if it doesnt stop the streamlit display
            else:
                st.sidebar.warning("No file uploaded.")
                st.stop()
        # if selection is not upload data
        else:
            return selection, None

    def show_kpi_func(self):
        col1, col2, col3, col4, col5 = st.columns(5)

    def show_graphs_func(self):
        pass

    def show_inventory_func(self):
        pass

    def show_ml_func(self):
        pass

    def show_download_func(self):
        pass