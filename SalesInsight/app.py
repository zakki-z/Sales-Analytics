from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = "restaurant_sales_analysis"


# Load data function
def load_data():
    try:
        sales_data = pd.read_csv("restaurant_sales.csv")
        return sales_data
    except FileNotFoundError:
        return None


# Categorize sales
def categorize_sales(sales):
    if sales > 3000:
        return 'Best Seller'
    elif 1000 <= sales <= 3000:
        return 'Moderate Seller'
    else:
        return 'Low Seller'


# Generate plot as base64 image
def create_figure_as_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


# Generate distribution plot
def generate_distribution_plot(sales_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=sales_data, x='SalesCategory', palette='viridis',
                  order=['Best Seller', 'Moderate Seller', 'Low Seller'], ax=ax)
    ax.set_title('Distribution des catégories de ventes', fontsize=16)
    ax.set_ylabel('Nombre de produits', fontsize=12)
    ax.set_xlabel('Catégorie de ventes', fontsize=12)
    return create_figure_as_base64(fig)


# Generate sales by category plot
def generate_category_plot(sales_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=sales_data, x='Category', y='Sales', palette='pastel', ax=ax)
    ax.set_title('Répartition des ventes par catégorie', fontsize=16)
    return create_figure_as_base64(fig)


# Generate prediction plot
def generate_prediction_plot(new_products):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=new_products, x='Sales', y='GrowthTrend',
                    hue='PredictedBestSeller', palette='coolwarm', s=150, ax=ax)
    ax.set_title('Prédictions des nouvelles tendances de vente', fontsize=16)
    ax.set_xlabel('Ventes (Simulées)', fontsize=12)
    ax.set_ylabel('Tendance de Croissance', fontsize=12)
    ax.legend(title='Meilleure Vente (1=Oui, 0=Non)', loc='upper left')
    return create_figure_as_base64(fig)


# Train model and make predictions
def train_model_and_predict(sales_data, new_product_data=None):
    sales_data['IsBestSeller'] = (sales_data['SalesCategory'] == 'Best Seller').astype(int)

    # Features and labels
    features = sales_data[['Sales', 'GrowthTrend']]
    labels = sales_data['IsBestSeller']

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Predict for new products if provided
    predictions_df = None
    predictions_plot = None

    if new_product_data is not None:
        try:
            # Convert form data to DataFrame
            new_products = pd.DataFrame(new_product_data)

            # Scale the data
            new_products_scaled = scaler.transform(new_products[['Sales', 'GrowthTrend']])

            # Make predictions
            predictions = model.predict(new_products_scaled)
            new_products['PredictedBestSeller'] = predictions

            # Generate plot
            predictions_plot = generate_prediction_plot(new_products)

            return {
                'accuracy': round(accuracy * 100, 2),
                'report': report,
                'new_products': new_products.to_dict('records'),
                'predictions_plot': predictions_plot
            }
        except Exception as e:
            return {
                'accuracy': round(accuracy * 100, 2),
                'report': report,
                'error': str(e)
            }

    return {
        'accuracy': round(accuracy * 100, 2),
        'report': report
    }


@app.route('/')
def index():
    sales_data = load_data()
    if sales_data is None:
        return render_template('error.html',
                               message="Fichier CSV non trouvé. Veuillez vous assurer que 'restaurant_sales.csv' est disponible.")

    # Process data
    sales_data['SalesCategory'] = sales_data['Sales'].apply(categorize_sales)

    # Generate plots
    distribution_plot = generate_distribution_plot(sales_data)
    category_plot = generate_category_plot(sales_data)

    # Get data summary
    data_summary = {
        'total_products': len(sales_data),
        'best_sellers': len(sales_data[sales_data['SalesCategory'] == 'Best Seller']),
        'moderate_sellers': len(sales_data[sales_data['SalesCategory'] == 'Moderate Seller']),
        'low_sellers': len(sales_data[sales_data['SalesCategory'] == 'Low Seller']),
        'avg_sales': round(sales_data['Sales'].mean(), 2),
        'max_sales': sales_data['Sales'].max(),
        'min_sales': sales_data['Sales'].min()
    }

    # Get model metrics
    model_metrics = train_model_and_predict(sales_data)

    return render_template('index.html',
                           data_summary=data_summary,
                           distribution_plot=distribution_plot,
                           category_plot=category_plot,
                           model_metrics=model_metrics,
                           sample_data=sales_data.head(10).to_dict('records'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sales_data = load_data()
        if sales_data is None:
            return render_template('error.html', message="Fichier CSV non trouvé.")

        sales_data['SalesCategory'] = sales_data['Sales'].apply(categorize_sales)

        # Get form data
        new_product_count = int(request.form.get('product_count', 1))
        new_product_data = {
            'Sales': [],
            'GrowthTrend': []
        }

        for i in range(1, new_product_count + 1):
            sales = float(request.form.get(f'sales_{i}', 0))
            growth = float(request.form.get(f'growth_{i}', 0))
            new_product_data['Sales'].append(sales)
            new_product_data['GrowthTrend'].append(growth)

        # Train model and predict
        prediction_results = train_model_and_predict(sales_data, new_product_data)

        return render_template('predict.html',
                               prediction_results=prediction_results,
                               product_count=new_product_count)
    else:
        return render_template('predict.html', product_count=5)


@app.route('/data')
def data():
    sales_data = load_data()
    if sales_data is None:
        return render_template('error.html', message="Fichier CSV non trouvé.")

    sales_data['SalesCategory'] = sales_data['Sales'].apply(categorize_sales)

    return render_template('data.html', data=sales_data.to_dict('records'))


if __name__ == '__main__':
    app.run(debug=True)
