from sklearn.model_selection import StratifiedKFold
import pybase64
from joblib import dump, load
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


warnings.filterwarnings('ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)

pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 500)


model = load('model.joblib')

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
# st.markdown('<style>body{background-color: #c9eb34;}</style>',unsafe_allow_html=True)

# Page Heading...
st.title("Health Insurance Claim Fraud Detection App")
st.subheader('@Author: Binita')

st.sidebar.title("User Options:")
page = st.sidebar.selectbox(
    "Please select from these",
    ("Download Sample file format",
     "Loaded file preview",
     "Visualize loaded data",
     "Predictive Modelling")
)


def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = pybase64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


file = st.file_uploader("Load your CSV file here", type="csv")

if page == "Loaded file preview":
    if file is not None:
        preview = pd.read_csv(file)
        st.write(preview.head())
    else:
        st.info('Please upload csv file.')

elif page == "Download Sample file format":
    df = pd.read_csv('sample_format.csv')
    tmp_download_link = download_link(
        df, 'sample_data_format.csv', 'Click here to download Sample file format!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

elif page == "Visualize loaded data":
    if file is not None:

        data = pd.read_csv(file)

        if 'Result' in data.columns:
            st.sidebar.subheader('Selct below options to visualize the graph')
            plots = st.sidebar.radio(label="", options=["Fraud Counts - Payment_Typology",
                                                        "Ratio distribution",
                                                        "Fraud Counts - Age",
                                                        "Fraud Counts - Gender"])

            if plots == "Fraud Counts - Payment_Typology":
                fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                st.subheader('Fraud details w.r.t. Payment_Typology')
                st.write(sns.countplot(data.Payment_Typology, hue=data.Result,))
                plt.xticks(rotation=45)
                st.pyplot()

            elif plots == "Ratio distribution":
                fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                st.subheader('Cost to Charge Ratio for fraud cases')
                st.write(sns.distplot(
                    data[data['Result'] == 0]['ratio_of_total_costs_to_total_charges']))
                st.pyplot()

                fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                st.subheader('Cost to Charge Ratio for Genuine cases')
                st.write(sns.distplot(
                    data[data['Result'] == 1]['ratio_of_total_costs_to_total_charges']))
                st.pyplot()

            elif plots == "Fraud Counts - Age":
                fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                st.subheader('Fraud details w.r.t. Age')
                st.write(sns.countplot(data.Age, hue=data.Result))
                plt.xticks(rotation=45)
                st.pyplot()

            elif plots == "Fraud Counts - Gender":
                fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                st.subheader('Fraud details w.r.t. Gender')
                st.write(sns.countplot(data.Gender, hue=data.Result))
                plt.xticks(rotation=45)
                st.pyplot()

        else:
            st.write(
                "Note: Visualization supported to datasets having 'Result' column.")
    else:
        st.info('Please upload csv file.')

elif page == "Predictive Modelling":
    if file is not None:

        data = pd.read_csv(file)

        dataset = data.copy()

        dataset.replace({'Days_spend_hsptl': '120 +'}, 120, inplace=True)
        dataset.Days_spend_hsptl = dataset.Days_spend_hsptl.astype('int')

        # drop null values & duplicates...
        dataset = dataset.dropna()
        # dataset.drop_duplicates(inplace=True)

        # removing space in column names...
        dataset.columns = dataset.columns.to_series().apply(
            lambda x: x.replace(' ', '_')).to_list()
        dataset.columns = dataset.columns.to_series().apply(
            lambda x: x.replace('/', '_')).to_list()
        dataset.rename(
            columns={'Home_or_self_care,': 'Home_or_self_care'}, inplace=True)

        dataset.drop(['Hospital_Id', 'apr_drg_description',
                      'Abortion', 'Weight_baby'], axis=1, inplace=True)

        # label encoding  one hot encoding...
        label_encoder_list = load('label_encoder_list.joblib')
        one_hot_coder_list = load('one_hot_coder_list.joblib')

        label_to_column = load('label_to_column.joblib')
        one_hot_column = load('one_hot_column.joblib')

        j = 0
        for i in label_to_column:
            dataset[i] = label_encoder_list[j].fit_transform(dataset[i])
            j = +1

        k = 0
        for i in one_hot_column:
            dataset[i] = one_hot_coder_list[k].fit_transform(dataset[i])
            k = +1

        if 'Result' in dataset.columns:

            prediction = model.predict(dataset.drop(['Result'], axis=1))

            st.radio(label="", options=[
                     "Download predicted result dataset!"])
            output_data = data.loc[dataset.index]

            output_data['Prediction'] = pd.DataFrame(prediction)
            output_data['Prediction'] = output_data.Prediction.apply(
                lambda x: 'Fraud' if x == 0 else 'Genuine')
            tmp_download_link1 = download_link(
                output_data, 'predicted_data.csv', 'Click here to download predicted result file!')
            st.markdown(tmp_download_link1, unsafe_allow_html=True)

            st.write('F1 Score :', f1_score(
                dataset.Result, prediction).round(2))
            st.write('Accuracy :', accuracy_score(
                dataset.Result, prediction).round(2))

            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
            st.subheader('Loaded data balance state..')
            st.write(sns.countplot(dataset.Result))
            st.pyplot()

            cm = confusion_matrix(dataset.Result, prediction)
            st.subheader('Confusion Matrix:')
            st.write(sns.heatmap(cm, annot=True, fmt='.8g', xticklabels=[
                     'Fraud', 'Genuine'], yticklabels=['Fraud', 'Genuine']))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot()

        else:

            st.radio(label="", options=[
                     "Download predicted result dataset!"])
            output_data = data.loc[dataset.index]
            prediction = model.predict(dataset)
            output_data['Prediction'] = pd.DataFrame(prediction)
            output_data['Prediction'] = output_data.Prediction.apply(
                lambda x: 'Fraud' if x == 0 else 'Genuine')
            tmp_download_link2 = download_link(
                output_data, 'predicted_data.csv', 'Click here to download predicted result file!')
            st.markdown(tmp_download_link2, unsafe_allow_html=True)
    else:
        st.info('Please upload csv file.')


st.sidebar.title('GitHub-Source Code:')
link = 'https://github.com/'Binita72/Fraud_Detection_ml_model_streamlit
st.sidebar.markdown(link, unsafe_allow_html=True)
