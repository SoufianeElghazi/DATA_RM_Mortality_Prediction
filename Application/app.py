from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Chargement du modèle et du préprocesseur
model = joblib.load('model.pkl')

# Colonnes que vous souhaitez inclure dans le formulaire
columns = ['age', 'gender', 'BMI', 'hypertensive', 'atrialfibrillation',
           'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression',
           'hyperlipidemia', 'Renal failure', 'COPD', 'heart rate',
           'Systolic blood pressure', 'Diastolic blood pressure',
           'Respiratory rate', 'temperature', 'SP O2', 'Urine output',
           'hematocrit', 'RBC', 'MCH', 'MCHC', 'MCV', 'RDW', 'Leucocyte',
           'Platelets', 'Neutrophils', 'Basophils', 'Lymphocyte', 'PT', 'INR',
           'NT-proBNP', 'Creatine kinase', 'Creatinine', 'Urea nitrogen',
           'glucose', 'Blood potassium', 'Blood sodium', 'Blood calcium',
           'Chloride', 'Anion gap', 'Magnesium ion', 'PH', 'Bicarbonate',
           'Lactic acid', 'PCO2', 'EF']

# Charger les données et calculer les valeurs min et max
patients = pd.read_csv('data.csv')
patients.drop(["group", "ID"], axis=1, inplace=True)
X_train = patients.drop('outcome', axis=1)
sc = StandardScaler()
sc.fit(X_train)

# Calculer les valeurs min et max pour chaque colonne
min_values = 0
max_values = patients.max().to_dict()

@app.route('/')
def home():
    # Ajouter cette ligne pour transmettre les colonnes, min_values, et max_values dans le contexte
    return render_template('index.html', columns=columns, min_values=min_values, max_values=max_values)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        data = []
        for column in columns:
            value = request.form[column]
            try:
                # Tenter de convertir la valeur en nombre flottant
                float_value = float(value)
                data.append(float_value)
            except ValueError:
                # Gérer le cas où la conversion échoue (par exemple, la valeur est une chaîne vide)
                return render_template('index.html', columns=columns, min_values=min_values, max_values=max_values, error="Veuillez saisir des valeurs valides.")

        # Appliquer la fonction de prétraitement
        sc = StandardScaler()
        processed_data = sc.fit_transform([data])

        # Prédiction
        prediction = model.predict(processed_data)
        if prediction == 0:
            prediction_text = "ce Patient est normal"
            prediction_image = "../static/assets/normal.jpg"
        else:
            prediction_text = "ce Patient est urgent"
            prediction_image = "../static/assets/urgent.jpg"

        return render_template('index.html', columns=columns, min_values=min_values, max_values=max_values,
                               prediction=prediction_text, prediction_image=prediction_image)

if __name__ == '__main__':
    app.run(debug=True)
