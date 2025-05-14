from treinar_modelo import treinar_modelo
import pandas as pd

tabela_train = pd.read_csv("meu_projeto_ia_docker/Data/bootcamp_train.csv")

treinar_modelo(tabela_train)