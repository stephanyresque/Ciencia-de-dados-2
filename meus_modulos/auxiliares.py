import pandas as pd

def tabela_dist_freq(dataframe, coluna):

    df_estatistica = pd.DataFrame()

    df_estatistica['frequencia'] = dataframe[coluna].value_counts().sort_index()

    df_estatistica['frequencia_relativa'] = dataframe[coluna].value_counts(normalize=True).sort_index()

    df_estatistica['frequencia_acumulada'] = df_estatistica['frequencia'].cumsum()

    df_estatistica['frequencia_relativa_acumulada'] = df_estatistica['frequencia_relativa'].cumsum()


    return df_estatistica
