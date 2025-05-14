import pandas as pd
import numpy as np

def limpar_dados(tabela) :


    tabela_train = tabela

    tabela_train[tabela_train.duplicated()]

    tabela_train = tabela_train.drop(["id", "y_minimo", "y_maximo", "peso_da_placa"], axis=1) 

    def remover_nulos(df, coluna) :
        df[coluna] = df[coluna].fillna(df[coluna].median())
        return df

    tabela_train = remover_nulos(tabela_train, "x_maximo")
    tabela_train = remover_nulos(tabela_train, "soma_da_luminosidade")
    tabela_train = remover_nulos(tabela_train, "maximo_da_luminosidade")
    tabela_train = remover_nulos(tabela_train, "espessura_da_chapa_de_aço")
    tabela_train = remover_nulos(tabela_train, "index_quadrado")
    tabela_train = remover_nulos(tabela_train, "indice_global_externo")
    tabela_train = tabela_train.dropna()

    tabela_train["x_maximo"] = tabela_train["x_maximo"].astype(int)
    tabela_train["maximo_da_luminosidade"] = tabela_train["maximo_da_luminosidade"].astype(int)
    tabela_train["soma_da_luminosidade"] = tabela_train["soma_da_luminosidade"].astype(int)
    tabela_train["espessura_da_chapa_de_aço"] = tabela_train["espessura_da_chapa_de_aço"].astype(int)

    def corrigir_true_false(df, coluna) :
        correcoes = {
        "0" : 0,
        "nao" : 0,
        "não" : 0,
        "Não" : 0,
        "FALSE" : 0,
        "False" : 0,
        False : 0,
        "N" : 0,
        "1" : 1,
        "S" : 1,
        "y" : 1,
        "Sim" : 1,
        "sim" : 1,
        "TRUE" : 1,
        "True" : 1,
        True : 1
        }

        df[coluna] = df[coluna].replace(correcoes)
        return df


    tabela_train = corrigir_true_false(tabela_train, "falha_1")
    tabela_train = corrigir_true_false(tabela_train, "falha_2")
    tabela_train = corrigir_true_false(tabela_train, "falha_3")
    tabela_train = corrigir_true_false(tabela_train, "falha_4")
    tabela_train = corrigir_true_false(tabela_train, "falha_5")
    tabela_train = corrigir_true_false(tabela_train, "falha_6")
    tabela_train = corrigir_true_false(tabela_train, "falha_outros")
    tabela_train = corrigir_true_false(tabela_train, "tipo_do_aço_A300")
    tabela_train = corrigir_true_false(tabela_train, "tipo_do_aço_A400")

    # Remover a linha que contém "-" pois não sabemos se é 0 ou 1
    tabela_train = tabela_train[tabela_train["tipo_do_aço_A300"] != "-"]

    # mudar o Dtype das colunas tipo do aço
    tabela_train["tipo_do_aço_A300"] = tabela_train["tipo_do_aço_A300"].astype(int)
    tabela_train["tipo_do_aço_A400"] = tabela_train["tipo_do_aço_A400"].astype(int)

    mediana = tabela_train["comprimento_do_transportador"].median()
    tabela_train["comprimento_do_transportador"] = tabela_train["comprimento_do_transportador"].mask(tabela_train["comprimento_do_transportador"] < 0, mediana)

    tabela_train = tabela_train.abs()

    colunas_falha = ["falha_1", "falha_2", "falha_3", "falha_4", "falha_5", "falha_6", "falha_outros"]

    tabela_train["falha"] = tabela_train[colunas_falha].idxmax(axis=1)
    tabela_train = tabela_train.drop(columns=colunas_falha)

    colunas_aco = ["tipo_do_aço_A300", "tipo_do_aço_A400"]

    tabela_train["tipo_aço"] = tabela_train[colunas_aco].idxmax(axis=1)
    tabela_train = tabela_train.drop(columns=colunas_aco)

    def remover_outliers_pelo_tipo_falha(df, coluna_outlier, coluna_falha) :
        df_sem_outliers = pd.DataFrame()

        for falha, tabela in df.groupby(coluna_falha) :
            Q1 = tabela[coluna_outlier].quantile(0.25)
            Q3 = tabela[coluna_outlier].quantile(0.75)
            IQR = Q3 - Q1

            limite_inferior = Q1 - 3 * IQR
            limite_superior = Q3 + 3 * IQR

            mediana = tabela[coluna_outlier].median()

            tabela[coluna_outlier] = np.where(
                (tabela[coluna_outlier] < limite_inferior) | (tabela[coluna_outlier] > limite_superior),
                mediana,
                tabela[coluna_outlier]
            )

            df_sem_outliers = pd.concat([df_sem_outliers, tabela])

        return df_sem_outliers

    numerico = [coluna for coluna in tabela_train.columns if tabela_train[coluna].nunique() > 10]

    for column in numerico :
        tabela_train = remover_outliers_pelo_tipo_falha(
            tabela_train,
            column,
            "falha"
        )

    return tabela_train