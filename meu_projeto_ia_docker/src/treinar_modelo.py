import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from data_carregar_limpeza import limpar_dados
from modelo_utils import modelo

def treinar_modelo(tabela) :
    tabela_train = tabela

    tabela_train = limpar_dados(tabela_train)

    numerico = [coluna for coluna in tabela_train.columns if tabela_train[coluna].nunique() > 10]

    X = tabela_train.drop(columns=["falha"]).copy()
    y = tabela_train["falha"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    y_test = le.transform(y_test)

    preprocessor = ColumnTransformer(
        transformers= [
            ("falha", StandardScaler(), numerico)
        ]
    )

    xgb_clf = Pipeline(steps=[
        ("processando", preprocessor),
        ("classificar", xgb.XGBClassifier(random_state=42, base_score=0.5))
    ])

    xgb_clf.fit(X_train, y_train)

    y_pred = xgb_clf.predict(X_test)

    y_numerico = le.fit_transform(y)
    cvs = cross_val_score(xgb_clf, X, y_numerico, cv=10, scoring="accuracy")

    modelo_treinado = modelo(xgb_clf, "meu_projeto_ia_docker\modelos_treinados\modelo_xgb.pkl")

    return modelo_treinado