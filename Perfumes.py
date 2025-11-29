from DataBase import perfumes
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import re


# FUNCIÓN PARA ENTENDER TEXTO LIBRE

def extraer_atributos(texto):
    texto = texto.lower()

    # Lista de TODOS los atributos posibles 
    atributos_posibles = set()
    for p in perfumes:
        atributos_posibles.update([p['familia'].lower(),
                                   p['clima'].lower(),
                                   p['ocasion'].lower(),
                                   p['genero'].lower()])
        atributos_posibles.update([n.lower() for n in p['notas_salida']])
        atributos_posibles.update([n.lower() for n in p['notas_corazon']])
        atributos_posibles.update([n.lower() for n in p['notas_fondo']])

    encontrados = []
    for atr in atributos_posibles:
        # Coincide aunque esté dentro de una frase
        if re.search(r"\b" + re.escape(atr) + r"\b", texto):
            encontrados.append(atr)

    return encontrados


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



# FUNCIÓN DE RECOMENDACIÓN HIBRIDA

def recomendar_perfume(texto_usuario, top_n=3):

    # EXTRAER ATRIBUTOS INTELIGENTES DEL TEXTO
    atributos_usuario = extraer_atributos(texto_usuario)

    if not atributos_usuario:
        return "No pude identificar las características."

    # Vectorizar entrada
    texto_vec = vectorizer.transform([texto_usuario.lower()])

    # Predicción (probabilidades)
    proba = clf_nn.predict_proba(texto_vec)[0]

    scores = []
    for _, row in df.iterrows():

        # Atributos que sí coinciden entre usuario y perfume
        atributos_coinciden = [a for a in atributos_usuario if a in row['atributos']]

        if not atributos_coinciden:
            continue

        # Índices para la red neural
        indices = [mlb.classes_.tolist().index(a) for a in atributos_coinciden]

        # Score del modelo (para ordenar)
        score_modelo = sum([proba[i] for i in indices]) / len(indices)

        # Porcentaje real por coincidencia exacta
        coincidencias = len(atributos_coinciden)
        total_usuario = len(atributos_usuario)
        porcentaje_real = round((coincidencias / total_usuario) * 100)

        scores.append((score_modelo, porcentaje_real, row))

    # Ordenar por mayor porcentaje real
    scores.sort(reverse=True, key=lambda x: x[1])

    # Construir respuesta
    resultados = ""
    for score_modelo, porcentaje_real, info in scores[:top_n]:
        resultados += f"""
Te recomiendo: '{info['nombre']}' de la marca {info['marca']} (Coincidencia: {porcentaje_real}%)

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


# Ejemplo
usuario = "me gustaría un perfume con ron y tabaco para clima frío"
print(recomendar_perfume(usuario))
