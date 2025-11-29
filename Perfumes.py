from DataBase import perfumes
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Preparar datos
df = pd.DataFrame(perfumes)

# Combinar atributos en un solo texto
df['texto_completo'] = (
    df['familia'] + " " + df['clima'] + " " + df['ocasion'] + " " +
    df['genero'] + " " +
    df['notas_salida'].apply(lambda x: " ".join(x)) + " " +
    df['notas_corazon'].apply(lambda x: " ".join(x)) + " " +
    df['notas_fondo'].apply(lambda x: " ".join(x))
)

# Etiquetas multi-label
df['atributos'] = df.apply(lambda row:
                           [row['familia'].lower(), row['clima'].lower(), row['ocasion'].lower(),
                            row['genero'].lower()] +
                           [n.lower() for n in row['notas_salida']] +
                           [n.lower() for n in row['notas_corazon']] +
                           [n.lower() for n in row['notas_fondo']], axis=1)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['atributos'])

# Vectorizar texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto_completo'])

# Entrenar red neuronal
clf_nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)
clf_nn.fit(X, y)

# Función de recomendación
def recomendar_perfume(texto_usuario, top_n=3):
    atributos_usuario = [a.strip().lower() for a in texto_usuario.split(",")]

    # Vectorizar entrada
    texto_vec = vectorizer.transform([texto_usuario.lower()])

    # Predicción 
    proba = clf_nn.predict_proba(texto_vec)[0]

    scores = []
    for _, row in df.iterrows():

        #  comparar atributos mencionados POR EL USUARIO
        atributos_validos = [a for a in atributos_usuario]

        # Filtrar  los atributos del perfume que coinciden con lo que dijo el usuario
        atributos_coinciden = [a for a in atributos_validos if a in row['atributos']]

        # Si el perfume no tiene ninguno de los atributos 
        if not atributos_coinciden:
            continue

        # Índices de atributos que sí coincide
        indices = [mlb.classes_.tolist().index(a) for a in atributos_coinciden]

        # Score del modelo 
        score_modelo = sum([proba[i] for i in indices]) / len(indices)

        # Porcentaje REAL basado en coincidencias exactas
        coincidencias = len(atributos_coinciden)
        total_usuario = len(atributos_usuario)
        porcentaje_real = round((coincidencias / total_usuario) * 100)

        # Guardamos ambos: score  y porcentaje real 
        scores.append((score_modelo, porcentaje_real, row))

    # Ordenar por score del modelo
    scores.sort(reverse=True, key=lambda x: x[1])

    # Recomendacion
    resultados = ""
    for score_modelo, porcentaje_real, info in scores[:top_n]:
        porcentaje = porcentaje_real

        resultados += f"""
Te recomiendo: '{info['nombre']}' de la marca {info['marca']} (Coincidencia: {porcentaje}%)

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

    return resultados


# modo de uso:
usuario = "frambuesa, mandarina"
print(recomendar_perfume(usuario))
