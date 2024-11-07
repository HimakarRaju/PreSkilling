import streamlit as st
import plotly.express as px
import pandas as pd
import sweetviz as sv
import ydata_profiling
import dtale
import os
from PIL import Image

# Set wide mode and light theme by default
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="Interactive Data Exploration Dashboard",  # Set page title
    page_icon=Image.open(r"C:\Projects\LOGO.png"),  # Set page icon
)

# Set default font to serif
st.markdown(
    """
    <style>
    body {
        font-family: serif;
        align-items: center;
    }
    *, ::before, ::after {
        box-sizing: border-box;
        text-align: center;
        justify-content: space-between;
    }
    .st-emotion-cache-1kyxreq {
        display: flex;
        flex-flow: wrap;
        row-gap: 1rem;
        justify-content: center;
        align-items: center;
    }
    .stApp {
        text-align: center; /* Center the title */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Interactive Data Exploration Dashboard")
# Display the logo image
st.image(Image.open(r"C:\Projects\LOGO.png"), width=80)

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Extract filename without extension
        filename_without_extension = os.path.splitext(uploaded_file.name)[0]

        # Create a folder named after the uploaded file
        folder_name = filename_without_extension
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Data Processing
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type in ("xlsx","xls"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type.")
            df = None

        if df is not None:
            # Display DataFrame
            st.subheader("Uploaded Data:")
            st.dataframe(df)

            # Graph Options
            st.subheader("Graph Options:")
            graph_type = st.selectbox("Select a graph type", ["Scatter", "Line", "Bar", "Pie"])
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            color = st.color_picker("Select color")

            # Dynamic Visualization
            if graph_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=[color])
            elif graph_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis, color_discrete_sequence=[color])
            elif graph_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis, color_discrete_sequence=[color])
            elif graph_type == "Pie":
                fig = px.pie(df, values=y_axis, names=x_axis, color_discrete_sequence=[color])

            # Display the plot
            st.plotly_chart(fig)

            # Save the plot
            fig.write_html(f"{folder_name}/{graph_type}_{x_axis}_{y_axis}.html")

            # --- Data Exploration and Transformation ---
            st.subheader("Data Exploration")

            # Describe
            if st.checkbox("Show Descriptive Statistics"):
                st.write(df.describe())

            # Duplicates
            if st.checkbox("Show Duplicates"):
                duplicates = df[df.duplicated()]
                st.write("Duplicates:")
                st.dataframe(duplicates)

            # Missing Analysis
            if st.checkbox("Show Missing Value Analysis"):
                st.write(df.isnull().sum())

            num_cols = df.select_dtypes(include="number")
            st.write(f"numerical columns : {num_cols.columns.to_list()}")
            # Correlations
            if st.checkbox("Show Correlation Matrix"):
                if isinstance(num_cols, pd.DataFrame):
                    st.write(num_cols.corr())
                else:
                    st.write("Not enough numerical columns")

            # Heat Map
            if st.checkbox("Generate Heatmap"):
                if isinstance(num_cols, pd.DataFrame):
                    fig = px.imshow(num_cols.corr())
                    st.plotly_chart(fig)
                else:
                    st.write("Not enough numerical columns")

            # Report Generation Options
            st.subheader("Generate Reports:")
            # Arranging the buttons
            col1, col2, col3 = st.columns(3)

            with col2:
                if st.button("Generate Sweetviz Report"):
                    with st.spinner("Generating Sweetviz report..."):
                        report = sv.analyze(df)
                        report.show_html(filepath=f"{folder_name}/sweetviz_report.html", layout='vertical',
                                         open_browser=True)

            with col1:
                if st.button("Generate ydata-profiling Report"):
                    with st.spinner("Generating ydata-profiling report..."):
                        profile = ydata_profiling.ProfileReport(df)
                        profile.to_file(f"{folder_name}/ydata_profiling_report.html")

                        # Open the report in Streamlit
                        with open(f"{folder_name}/ydata_profiling_report.html", "r") as f:
                            html_string = f.read()
                        st.components.v1.html(html_string, width=1800, height=1200,  scrolling=True)

            with col3:
                if st.button("Launch dtale"):
                    with st.spinner("Launching dtale..."):
                        d = dtale.show(df)
                        d.open_browser()

    except Exception as e:
        st.error(f"An error occurred: {e}")
