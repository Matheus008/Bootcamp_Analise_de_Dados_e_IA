import joblib

def modelo(modelo, caminho) :
    joblib.dump(modelo, caminho)
    print(f"modelo salvo no caminho: {caminho}")