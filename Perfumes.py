from DataBase import perfumes
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Preparar datos

df = pd.DataFrame(perfumes)

# Combinar atributos en texto
df['texto_completo'] = (
    df['familia'] + " " + df['clima'] + " " + df['ocasion'] + " " +
    df['genero'] + " " + df['nombre'] + " " + df['marca'] + " " +
    df['notas_salida'].apply(lambda x: " ".join(x)) + " " +
    df['notas_corazon'].apply(lambda x: " ".join(x)) + " " +
    df['notas_fondo'].apply(lambda x: " ".join(x))
)

# Vectorizar texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto_completo'])
y = df['nombre']


# Red neuronal simple 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf_nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
clf_nn.fit(X_train, y_train)
y_pred = clf_nn.predict(X_test)
print(f"Exactitud de la red neuronal: {accuracy_score(y_test, y_pred)*100:.2f}%\n")



# Función de recomendación

def recomendar_perfume(texto_usuario):
    texto_vec = vectorizer.transform([texto_usuario])
    perfume_recomendado = clf_nn.predict(texto_vec)[0]
    info = df[df['nombre'] == perfume_recomendado].iloc[0]

    detalles = f"""
Te recomiendo: '{info['nombre']}' de la marca {info['marca']}

Familia: {info['familia']}
Clima: {info['clima']}
Ocasión: {info['ocasion']}
Género: {info['genero']}
Intensidad: {info['intensidad']}
Duración: {info['duracion']} horas

Notas:
- Salida: {', '.join(info['notas_salida'])}
- Corazón: {', '.join(info['notas_corazon'])}
- Fondo: {', '.join(info['notas_fondo'])}
"""
    return detalles





# Prueba de recomendación

usuario = "Me gustaría un perfume que sea frutal"
print("=== Recomendación de la red neuronal ===")
print(recomendar_perfume(usuario))
