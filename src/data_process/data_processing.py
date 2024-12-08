import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from dash import html, dcc, dash_table
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def initial_eda(sales_data):
    buffer = []
    buffer.append(html.P(f'DataFrame Shape: {sales_data.shape}'))

    info = sales_data.dtypes.to_frame('Data Type')
    info['Data Type'] = info['Data Type'].astype(str)
    info['Non-Null Count'] = sales_data.notnull().sum()
    buffer.append(dash_table.DataTable(
        data=info.reset_index().to_dict('records'),
        columns=[{'name': i, 'id': i} for i in info.reset_index().columns],
        style_table={'overflowX': 'auto'}
    ))

    null_counts = sales_data.isnull().sum()
    buffer.append(html.H4('Null Values in Each Column'))
    buffer.append(dash_table.DataTable(
        data=null_counts.reset_index().rename(columns={'index': 'Column', 0: 'Null Count'}).to_dict('records'),
        columns=[{'name': 'Column', 'id': 'Column'}, {'name': 'Null Count', 'id': 'Null Count'}],
        style_table={'overflowX': 'auto'}
    ))

    duplicates = sales_data.duplicated().sum()
    buffer.append(html.P(f'Duplicate Rows in Dataset: {duplicates}'))

    summary = sales_data.describe()
    buffer.append(html.H4('Statistical Summary'))
    buffer.append(dash_table.DataTable(
        data=summary.reset_index().to_dict('records'),
        columns=[{'name': i, 'id': i} for i in summary.reset_index().columns],
        style_table={'overflowX': 'auto'}
    ))

    # Show missing value percentages
    missing_percent = (sales_data.isnull().sum() / len(sales_data)) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    if not missing_percent.empty:
        fig_missing_bar = px.bar(missing_percent, x=missing_percent.values, y=missing_percent.index,
                                 orientation='h', title='Percentage of Missing Values', text=missing_percent.values)
        fig_missing_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_missing_bar.update_layout(xaxis_title='Percentage', yaxis_title='Feature')
        buffer.append(dcc.Graph(figure=fig_missing_bar))

    return html.Div(buffer)

def generate_pre_cleaning_plots(sales_data):
    plots = []

    # Distribution of Item Fat Content
    fig1 = px.histogram(sales_data, x='Item_Fat_Content', title='Item Fat Content Distribution', text_auto=True)
    fig1.update_layout(xaxis_title='Item Fat Content', yaxis_title='Count')
    plots.append(dcc.Graph(figure=fig1))

    # Distribution of Item Visibility
    fig2 = px.histogram(sales_data, x='Item_Visibility', nbins=50, title='Item Visibility Distribution', text_auto=True)
    fig2.update_layout(xaxis_title='Item Visibility', yaxis_title='Count')
    plots.append(dcc.Graph(figure=fig2))

    # Missing Values Heatmap
    missing_data = sales_data.isnull()
    fig3 = px.imshow(missing_data.T, title='Missing Values Heatmap', color_continuous_scale='Blues', aspect='auto')
    plots.append(dcc.Graph(figure=fig3))

    return plots

def generate_advanced_pre_cleaning_plots(sales_data):
    plots = []
    numeric_cols = sales_data.select_dtypes(include=np.number).columns.tolist()
    if 'Item_Outlet_Sales' in numeric_cols:
        numeric_cols.remove('Item_Outlet_Sales')

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        corr = sales_data[numeric_cols + ['Item_Outlet_Sales']].corr()
        fig_corr = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
        plots.append(dcc.Graph(figure=fig_corr))

    # Scatter Matrix (Pairwise plots)
    selected_numeric = numeric_cols[:5] + ['Item_Outlet_Sales'] if len(numeric_cols) > 5 else numeric_cols + ['Item_Outlet_Sales']
    fig_pair = px.scatter_matrix(sales_data, dimensions=selected_numeric, title='Scatter Matrix of Selected Numeric Features')
    plots.append(dcc.Graph(figure=fig_pair))

    # Boxplot for Outlier Detection in the target (if present)
    if 'Item_Outlet_Sales' in sales_data.columns:
        fig_box = px.box(sales_data, y='Item_Outlet_Sales', title='Boxplot of Item_Outlet_Sales')
        plots.append(dcc.Graph(figure=fig_box))

    # Distribution of Target (Sales) with KDE
    if 'Item_Outlet_Sales' in sales_data.columns:
        fig_sales_dist = px.histogram(sales_data, x='Item_Outlet_Sales', nbins=50, title='Item_Outlet_Sales Distribution with KDE', marginal='violin', text_auto=True)
        fig_sales_dist.update_layout(xaxis_title='Item_Outlet_Sales', yaxis_title='Count')
        plots.append(dcc.Graph(figure=fig_sales_dist))

    return plots

def perform_data_cleaning(sales_data, log_transform_target=False, outlier_cap=True):

    # Fill missing Item_Weight
    Item_Weight_Mean = sales_data.pivot_table(values='Item_Weight', index='Item_Identifier', aggfunc='mean')
    item_weight_map = Item_Weight_Mean['Item_Weight'].to_dict()
    overall_mean_weight = sales_data["Item_Weight"].mean()
    sales_data["Item_Weight"] = sales_data["Item_Weight"].fillna(sales_data["Item_Identifier"].map(item_weight_map)).fillna(overall_mean_weight)

    # Fill missing Outlet_Size
    outlet_size_mode = sales_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan))
    missing_bool = sales_data['Outlet_Size'].isna()
    sales_data.loc[missing_bool, 'Outlet_Size'] = sales_data.loc[missing_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

    # Replace zeros in Item_Visibility with mean
    item_visibility_overall_mean = sales_data["Item_Visibility"].replace(0, np.nan).mean()
    sales_data["Item_Visibility"] = sales_data["Item_Visibility"].replace(0, item_visibility_overall_mean)

    # Standardize Item_Fat_Content
    sales_data["Item_Fat_Content"] = sales_data["Item_Fat_Content"].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

    # Add Product_Category
    sales_data["Product_Category"] = sales_data["Item_Identifier"].str[:2]
    sales_data["Product_Category"] = sales_data["Product_Category"].replace({'FD': 'Food Item', 'NC': 'Non Consumable', 'DR': 'Drink'})

    # Reclassify Non-Consumables
    sales_data.loc[sales_data['Product_Category'] == 'Non Consumable', 'Item_Fat_Content'] = 'Non Edible'

    # Calculate Outlet Age
    current_year = datetime.datetime.now().year
    sales_data['Outlet_Years'] = current_year - sales_data['Outlet_Establishment_Year']

    # Additional Cleaning: Outlier capping for target (e.g., 99th percentile)
    if 'Item_Outlet_Sales' in sales_data.columns and outlier_cap:
        cap_val = sales_data['Item_Outlet_Sales'].quantile(0.99)
        sales_data['Item_Outlet_Sales'] = np.where(sales_data['Item_Outlet_Sales'] > cap_val, cap_val, sales_data['Item_Outlet_Sales'])

    # Optional: Log transform target if desired
    if log_transform_target and 'Item_Outlet_Sales' in sales_data.columns:
        sales_data['Item_Outlet_Sales'] = np.log1p(sales_data['Item_Outlet_Sales'])

    return sales_data

def generate_post_cleaning_plots(sales_data):
    plots = []

    # Distribution of Item Fat Content after cleaning
    fig1 = px.histogram(sales_data, x='Item_Fat_Content', title='Item Fat Content Distribution (After Cleaning)', text_auto=True)
    fig1.update_layout(xaxis_title='Item Fat Content', yaxis_title='Count')
    plots.append(dcc.Graph(figure=fig1))

    # Distribution of Item Visibility after replacements
    fig2 = px.histogram(sales_data, x='Item_Visibility', nbins=50, title='Item Visibility Distribution (After Cleaning)', text_auto=True)
    fig2.update_layout(xaxis_title='Item Visibility', yaxis_title='Count')
    plots.append(dcc.Graph(figure=fig2))

    # Missing Values Heatmap after cleaning
    missing_data = sales_data.isnull()
    fig3 = px.imshow(missing_data.T, title='Missing Values Heatmap (After Cleaning)', color_continuous_scale='Blues', aspect='auto')
    plots.append(dcc.Graph(figure=fig3))

    # Counts of categories
    for col in ['Item_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Product_Category']:
        if col in sales_data.columns:
            counts = sales_data[col].value_counts().reset_index()
            counts.columns = [col, 'Count']
            fig = px.bar(counts, x=col, y='Count', title=f'Count of {col}', text='Count', labels={col: col, 'Count': 'Count'})
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_xaxes(tickangle=45)
            fig.update_layout(yaxis_title='Count')
            plots.append(dcc.Graph(figure=fig))

    # Distribution of Item Weight
    fig9 = px.histogram(
        sales_data,
        x='Item_Weight',
        nbins=20,
        title='Distribution of Item Weight (After Cleaning)',
        labels={'Item_Weight': 'Item Weight'},
        marginal="violin",
        text_auto=True
    )
    fig9.update_layout(yaxis_title='Count')
    plots.append(dcc.Graph(figure=fig9))

    return plots

def generate_advanced_post_cleaning_plots(original_data, cleaned_data):
    plots = []
    # Compare distributions before and after cleaning for Item_Outlet_Sales
    if 'Item_Outlet_Sales' in original_data.columns and 'Item_Outlet_Sales' in cleaned_data.columns:
        df_compare = pd.DataFrame({
            'Original': original_data['Item_Outlet_Sales'],
            'Cleaned': cleaned_data['Item_Outlet_Sales']
        })
        fig_compare = px.histogram(df_compare, barmode='overlay', text_auto=True)
        fig_compare.update_layout(title='Item_Outlet_Sales Distribution Before and After Cleaning', xaxis_title='Item_Outlet_Sales', yaxis_title='Count')
        fig_compare.update_traces(opacity=0.6)
        plots.append(dcc.Graph(figure=fig_compare))

    # Correlation Heatmap after cleaning
    numeric_cols = cleaned_data.select_dtypes(include=np.number).columns.tolist()
    if 'Item_Outlet_Sales' in numeric_cols:
        corr = cleaned_data[numeric_cols].corr()
        fig_corr_after = px.imshow(corr, text_auto=True, title='Correlation Heatmap (After Cleaning)')
        plots.append(dcc.Graph(figure=fig_corr_after))

    # Boxplot after outlier capping
    if 'Item_Outlet_Sales' in cleaned_data.columns:
        fig_box_after = px.box(cleaned_data, y='Item_Outlet_Sales', title='Boxplot of Item_Outlet_Sales (After Outlier Treatment)')
        plots.append(dcc.Graph(figure=fig_box_after))

    # Scatter matrix after cleaning
    selected_numeric = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
    if 'Item_Outlet_Sales' in numeric_cols and len(selected_numeric) > 1:
        fig_pair_after = px.scatter_matrix(cleaned_data, dimensions=selected_numeric, title='Scatter Matrix After Cleaning')
        plots.append(dcc.Graph(figure=fig_pair_after))

    return plots

def preprocess_data(sales_data):
    # Label Encoding
    label_encoder = LabelEncoder()
    sales_data["Outlet"] = label_encoder.fit_transform(sales_data["Outlet_Identifier"])

    # Encode categorical variables
    categorical_columns = sales_data.select_dtypes(include=['object']).columns.tolist()
    if 'Item_Identifier' in categorical_columns:
        categorical_columns.remove('Item_Identifier')
    if 'Outlet_Identifier' in categorical_columns:
        categorical_columns.remove('Outlet_Identifier')

    for category in categorical_columns:
        sales_data[category] = label_encoder.fit_transform(sales_data[category])

    # One-Hot Encoding for important categorical columns
    cols_to_dummy = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Product_Category']
    for c in cols_to_dummy:
        if c in sales_data.columns:
            sales_data = pd.get_dummies(sales_data, columns=[c], drop_first=False)

    # Features and target
    X = sales_data.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Item_Outlet_Sales'], axis=1)
    y = sales_data['Item_Outlet_Sales']
    feature_names = X.columns

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    return X_train, X_test, y_train, y_test, feature_names
