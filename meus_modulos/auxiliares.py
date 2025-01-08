import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene

def tabela_dist_freq(dataframe, coluna, coluna_frequencia = False):

    df_estatistica = pd.DataFrame()

    if coluna_frequencia:
        df_estatistica['frequencia'] = dataframe[coluna]
        df_estatistica['frequencia_relativa'] = df_estatistica['frequencia'] / df_estatistica['frequencia'].sum()
    else:
        df_estatistica['frequencia'] = dataframe[coluna].value_counts().sort_index()

        df_estatistica['frequencia_relativa'] = dataframe[coluna].value_counts(normalize=True).sort_index()

    df_estatistica['frequencia_acumulada'] = df_estatistica['frequencia'].cumsum()

    df_estatistica['frequencia_relativa_acumulada'] = df_estatistica['frequencia_relativa'].cumsum()


    return df_estatistica

def hist_box(dataframe, coluna, intervalos = 'auto'):

    fig, (ax1, ax2) = plt.subplots(
        nrows = 2, 
        ncols = 1, 
        sharex=True,
        gridspec_kw={
            'height_ratios': (0.15, 0.85),
            'hspace': 0.02
        }
    )
    
    sns.boxplot(
        data = dataframe, 
        x = coluna, 
        showmeans = True, 
        ax = ax1, 
        meanline = True,
        meanprops = {'color': 'C1', 'linewidth': 1.5},
        medianprops = {'color': 'C2', 'linewidth': 1.5, 'linestyle':'--'},
    )

    sns.histplot(data = dataframe, x = coluna, kde = True, bins = intervalos, ax = ax2)

    for ax in (ax1, ax2):
        ax.grid(True, linestyle = '--', color = 'gray', alpha = 0.5)
        ax.set_axisbelow(True)

    ax2.axvline(dataframe[coluna].mean(), color = 'C1', linestyle = '--', label = 'Média')
    ax2.axvline(dataframe[coluna].median(), color = 'C2', linestyle = '--', label = 'Mediana')
    ax2.axvline(dataframe[coluna].mode()[0], color = 'C3', linestyle = '--', label = 'Moda')
    ax2.legend()

    plt.show()

def analise_shapiro(dataframe, alfa = 0.05):
    print("Teste de Shapiro-Wilk")

    for coluna in dataframe.columns:
        estatistica_sw, valor_p_sw = shapiro(dataframe[coluna], nan_policy = 'omit')
        print(f"{estatistica_sw=:.3f}")
        if valor_p_sw > alfa:
            print(f"{coluna} segue uma distribuição normal (valor p: {valor_p_sw=:.3f})")
        else:
            print(f"{coluna} não segue uma distribuição normal (valor p: {valor_p_sw=:.3f})")


def analise_levene(dataframe, alfa = 0.05, center = 'mean'):
    print("Teste de Levene")

    estatistica_levene, valor_p_lv = levene(
        *[dataframe[coluna] for coluna in dataframe.columns], 
        center = center,
        nan_policy = 'omit'
    )

    print(f"{estatistica_levene=:.3f}")
    if valor_p_lv > alfa:
        print(f"Variâncias iguais (valor p: {valor_p_lv=:.3f})")
    else:
        print(f"Ao menos uma variância diferente (valor p: {valor_p_lv=:.3f})")

def analises_shapiro_levene(dataframe, alfa = 0.05, center= 'mean'):
    analise_shapiro(dataframe, alfa)

    print()

    analise_levene(dataframe, alfa, center)