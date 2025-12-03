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

def extraer_atributos(texto):
    texto = texto.lower()

    # Lista de TODOS los atributos posibles :3
    atributos_posibles = set()
    for p in perfumes:
        atributos_posibles.update([p['familia'].lower(),
                                   p['clima'].lower(),
                                   p['ocasion'].lower(),
                                   p['genero'].lower(),
                                   p['intensidad'].lower(),])
        atributos_posibles.update([n.lower() for n in p['notas_salida']])
        atributos_posibles.update([n.lower() for n in p['notas_corazon']])
        atributos_posibles.update([n.lower() for n in p['notas_fondo']])

    encontrados = []
    for atr in atributos_posibles:
        # Coincide aunque esté dentro de una palabra o frase
        if atr in texto:
            encontrados.append(atr)

    # Eliminar duplicados
    return list(set(encontrados))


# FUNCIÓN DE RECOMENDACIÓN HIBRIDA (coincidencia parcial)
def recomendar_perfume(texto_usuario, top_n=3):
    atributos_usuario = extraer_atributos(texto_usuario)

    if not atributos_usuario:
        return "No pude identificar las características."

    texto_vec = vectorizer.transform([texto_usuario.lower()])
    proba = clf_nn.predict_proba(texto_vec)[0]

    scores = []
    for _, row in df.iterrows():
        atributos_coinciden = []
        for a in atributos_usuario:
            for attr_perfume in row['atributos']:
                # Coincidencia parcial
                if a in attr_perfume or attr_perfume in a:
                    atributos_coinciden.append(a)
                    break

        if not atributos_coinciden:
            continue

        indices = [mlb.classes_.tolist().index(a) for a in atributos_coinciden if a in mlb.classes_]
        score_modelo = sum([proba[i] for i in indices]) / len(indices) if indices else 0

        coincidencias = len(atributos_coinciden)
        total_usuario = len(atributos_usuario)
        porcentaje_real = round((coincidencias / total_usuario) * 100)

        scores.append((score_modelo, porcentaje_real, row))

    # Ordenar por mayor porcentaje real
    scores.sort(reverse=True, key=lambda x: x[1])

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


# -------------------- Ejemplo --------------------
# usuario = "me gustaría un perfume lirio y madera"
# print(recomendar_perfume(usuario))

import tkinter as tk
from tkinter import scrolledtext, font

# Crear ventana principal
root = tk.Tk()
root.title("Recomendador de Perfumes")
root.geometry("800x600")
root.configure(bg="#f4f4f9")  # Fondo suave

# Fuente personalizada
titulo_font = font.Font(family="Helvetica", size=20, weight="bold")
label_font = font.Font(family="Helvetica", size=12)
texto_font = font.Font(family="Helvetica", size=11)

# Frame superior para título
frame_titulo = tk.Frame(root, bg="#f4f4f9")
frame_titulo.pack(pady=20)
titulo = tk.Label(frame_titulo, text="✨ Recomendador de Perfumes ✨", font=titulo_font, bg="#f4f4f9", fg="#4b3e72")
titulo.pack()

# Frame para entrada de usuario
frame_entrada = tk.Frame(root, bg="#e0dff6", bd=2, relief="ridge")
frame_entrada.pack(pady=10, padx=20, fill="x")

label = tk.Label(frame_entrada, text="Describe tus preferencias de perfume:", font=label_font, bg="#e0dff6")
label.pack(pady=5)

entrada = scrolledtext.ScrolledText(frame_entrada, wrap=tk.WORD, width=80, height=5, font=texto_font, bd=1, relief="solid")
entrada.pack(padx=10, pady=10)

# Frame para botón
frame_boton = tk.Frame(root, bg="#f4f4f9")
frame_boton.pack(pady=10)

def mostrar_recomendacion():
    texto_usuario = entrada.get("1.0", tk.END).strip()
    if texto_usuario:
        resultado = recomendar_perfume(texto_usuario, top_n=3)
        salida.config(state='normal')
        salida.delete("1.0", tk.END)
        salida.insert(tk.END, resultado)
        salida.config(state='disabled')

boton = tk.Button(frame_boton, text="🔍 Recomendar Perfumes", font=label_font, bg="#6c63ff", fg="white",
                  activebackground="#5751d1", activeforeground="white", padx=20, pady=10,
                  command=mostrar_recomendacion, relief="raised", bd=3)
boton.pack()

# Frame para salida de resultados
frame_salida = tk.Frame(root, bg="#f4f4f9")
frame_salida.pack(pady=10, padx=20, fill="both", expand=True)

salida = scrolledtext.ScrolledText(frame_salida, wrap=tk.WORD, width=90, height=20, font=texto_font, bd=2, relief="sunken")
salida.pack(padx=10, pady=10, fill="both", expand=True)
salida.config(state='disabled')

# Ejecutar ventana
root.mainloop()
