import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error("Error occurred while loading data: ", str(e))
        return None


def clean_column_names(data):
    data.columns = data.columns.str.replace("[^a-zA-Z0-9_]+", "_")
    return data


def handle_missing_values_and_encode(data):
    try:
        # Identify numeric and categorical columns
        numeric_features = data.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = data.select_dtypes(include=["object"]).columns

        # Define a pipeline for numeric imputation and categorical one-hot encoding with imputation
        preprocessing_pipeline = [
            ("numerical_imputation", SimpleImputer(strategy="mean"), numeric_features),
            (
                "categorical_imputation",
                SimpleImputer(strategy="most_frequent"),
                categorical_features,
            ),
            (
                "one_hot_encoding",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]

        column_transformer = ColumnTransformer(
            preprocessing_pipeline, remainder="passthrough"
        )

        # Fit and transform the data
        transformed_data = column_transformer.fit_transform(data)

        # Get names of encoded columns for one-hot encoding
        encoded_feature_names = column_transformer.named_transformers_[
            "one_hot_encoding"
        ].get_feature_names_out(categorical_features)
        transformed_column_names = list(numeric_features) + list(encoded_feature_names)

        # Create a mapping dictionary for one-hot encoding interpretation
        one_hot_mapping = {
            col: list(data[col].unique()) for col in categorical_features
        }

        # Create a DataFrame with the appropriate column names
        transformed_data = pd.DataFrame(
            transformed_data, columns=transformed_column_names
        )

        return transformed_data, one_hot_mapping
    except Exception as e:
        st.error("Error occurred while handling missing values and encoding: " + str(e))
        return None, None


def create_preprocessing_pipeline():
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    pipeline = Pipeline([("imputation", imputer), ("scaling", scaler)])

    return pipeline


def suggest_model(data):
    numeric_features = data.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = data.select_dtypes(include=["object"]).columns

    if len(numeric_features) > 0:
        model_options = [
            ("Random Forest Classifier", RandomForestClassifier()),
            ("Naive Bayes Classifier", GaussianNB()),
            ("Support Vector Machine Classifier", SVC()),
        ]
    else:
        model_options = [("Naive Bayes Classifier", GaussianNB())]

    return model_options


def main_function():
    file_path = st.file_uploader("Upload CSV file", type=["csv"])
    if file_path is not None:
        data = load_data(file_path)

        if data is not None:
            # Initial display and column management
            st.write("Original Data:")
            st.write(data.head())

            # Column renaming and type conversion (same as before)

            # Confirmation before continuing
            if st.button("Confirm and Continue"):
                clean_data = clean_column_names(data)
                st.write("Head of cleaned data:")
                st.write(clean_data.head())

                cleaned_data, one_hot_mapping = handle_missing_values_and_encode(
                    clean_data
                )

                if cleaned_data is not None:
                    preprocessing_pipeline = create_preprocessing_pipeline()
                    scaled_data = preprocessing_pipeline.fit_transform(cleaned_data)

                    st.write("Head of scaled data:")
                    st.write(pd.DataFrame(scaled_data).head())

                    # Display one-hot encoding mappings for reference
                    st.write("One-Hot Encoding Mappings:")
                    st.write(one_hot_mapping)

                    target_column = st.selectbox("Select target column", data.columns)
                    model_options = suggest_model(data)
                    selected_model_name, selected_model = st.selectbox(
                        "Select model", model_options
                    )

                    if selected_model is not None:
                        X = scaled_data
                        y = data[target_column]

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        model = selected_model
                        model.fit(X_train, y_train)

                        y_pred = model.predict(X_test)

                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average="weighted")
                        recall = recall_score(y_test, y_pred, average="weighted")
                        f1 = f1_score(y_test, y_pred, average="weighted")

                        st.write("Model performance metrics:")
                        st.write("Accuracy:", accuracy)
                        st.write("Precision:", precision)
                        st.write("Recall:", recall)
                        st.write("F1-score:", f1)

                        input_value = st.text_input("Enter input value:")
                        if input_value is not None:
                            try:
                                prediction = model.predict([[float(input_value)]])
                                st.write("Prediction:", prediction[0])
                            except Exception as e:
                                st.error("Error in prediction: " + str(e))


if __name__ == "__main__":
    main_function()
