from flask import Flask, request, jsonify
import joblib
import pandas as pd
from functions import preprocessing

# Inicializar la app Flask
app = Flask(__name__)

# Cargar el modelo
modelo = joblib.load('modelo_entrenado.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/api/predict', methods=['POST'])
def predecir():
    # Obtener datos JSON del request
    datos = request.get_json()
    
    # Convertir los datos a un DataFrame de pandas
    # Supongamos que los datos contienen características como: edad, ingresos, historial_crediticio, etc.
    df = pd.DataFrame([datos])
    df = preprocessing(df)
    print(df, flush=True)

    # Realizar la predicción
    prediccion = modelo.predict(df)
    
    print(prediccion, flush=True)
    # Devolver la respuesta en formato JSON
    resultado = {   
                 'prediction': bool(prediccion[0]),
                 'probability': float(prediccion[0])
                 }
    print(resultado, flush=True)  
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
