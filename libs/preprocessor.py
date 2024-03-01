# Módulo de tratamento de outliers
# Imports de bibliotecas
import numpy as np
import pandas as pd


# Classe
class Outlier:

    # Construtor
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        

    def count_outliers(self, Q1, Q3, IQR, columns):
        cut_off = IQR * 1.5
        temp_df = (self.df[columns] < (Q1 - cut_off)) | (self.df[columns] > (Q3 + cut_off))
        return [len(temp_df[temp_df[col] == True]) for col in temp_df]

    def calc_skew(self, columns=None):
        if columns == None:
            columns = self.df.columns
        return [self.df[col].skew() for col in columns]

    def percentage(self, list):
        return [str(round(((value/150001) * 100), 2)) + '%' for value in list]

    def remove_outliers(self, columns):
        for col in columns:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            cut_off = IQR * 1.5
            lower, upper = Q1 - cut_off, Q3 + cut_off
            self.df = self.df.drop(self.df[self.df[col] > upper].index)
            self.df = self.df.drop(self.df[self.df[col] < lower].index)

    def replace_outliers_with_fences(self, columns):
        for col in columns:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            cut_off = IQR * 1.5
            lower, upper = Q1 - cut_off, Q3 + cut_off

            self.df[col] = np.where(self.df[col] > upper, upper, self.df[col])
            self.df[col] = np.where(self.df[col] < lower, lower, self.df[col])

    def getOverview(self, columns) -> None:
        min = self.df[columns].min()
        Q1 = self.df[columns].quantile(0.25)
        median = self.df[columns].quantile(0.5)
        Q3 = self.df[columns].quantile(0.75)
        max = self.df[columns].max()
        IQR = Q3 - Q1
        skew = self.calc_skew(columns)
        outliers = self.count_outliers(Q1, Q3, IQR, columns)
        cut_off = IQR * 1.5
        lower, upper = Q1 - cut_off, Q3 + cut_off

        new_columns = ['Nome de Coluna', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'IQR', 'Lower fence', 'Upper fence', 'Skew', 'Num_Outliers', 'Percent_Outliers' ]
        
        data = zip([column for column in self.df[columns]], min, Q1, median, Q3, max, IQR, lower, upper, skew, outliers, self.percentage(outliers))

        new_df = pd.DataFrame(data = data, columns = new_columns)
        
        new_df.set_index('Nome de Coluna', inplace = True)
        \
        return new_df.sort_values('Num_Outliers', ascending = False).transpose()


########### Limpeza dos Dados
    
# Calcula o percentual de valores ausentes
def func_calc_percentual_valores_ausentes(df):

    # Calcula o total de células no dataset
    totalCells = np.product(df.shape)

    # Conta o número de valores ausentes por coluna
    missingCount = df.isnull().sum()

    # Calcula o total de valores ausentes
    totalMissing = missingCount.sum()

    # Calcula o percentual de valores ausentes
    print("O dataset tem", round(((totalMissing/totalCells) * 100), 2), "%", "de valores ausentes.")


# Função que calcula o percentual de linhas com valores ausentes
def func_calc_percentual_valores_ausentes_linha(df):

    # Calcula o número total de linhas com valores ausentes
    missing_rows = sum([True for idx,row in df.iterrows() if any(row.isna())])

    # Calcula o número total de linhas
    total_rows = df.shape[0]

    # Calcula a porcentagem de linhas ausentes
    print(round(((missing_rows/total_rows) * 100), 2), "%", "das linhas no conjunto de dados contêm pelo menos um valor ausente.")


# Função para calcular valores ausentes por coluna
def func_calc_percentual_valores_ausentes_coluna(df):
    
    # Total de valores ausentes
    mis_val = df.isnull().sum()

    # Porcentagem de valores ausentes
    mis_val_percent = 100 * mis_val / len(df)

    # Tipo de dado das colunas com valores ausentes
    mis_val_dtype = df.dtypes

    # Cria uma tabela com os resultados
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Renomear as colunas
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Valores Ausentes', 1 : '% de Valores Ausentes', 2: 'Dtype'})

    # Classifica a tabela por porcentagem de valores ausentes de forma decrescente e remove colunas sem valores faltantes
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,0] != 0].sort_values('% de Valores Ausentes', ascending = False).round(2)

    # Print 
    print ("O dataset tem " + str(df.shape[1]) + " colunas.\n"
        "Encontrado: " + str(mis_val_table_ren_columns.shape[0]) + " colunas que têm valores ausentes.")

    if mis_val_table_ren_columns.shape[0] == 0:
        return

    # Retorna o dataframe com informações ausentes
    return mis_val_table_ren_columns


# Imputação de valores ausentes usando forward fill (preenchimento progressivo)
# method = 'ffill': Ffill ou forward-fill propaga o último valor não nulo observado para frente até que outro valor não nulo seja encontrado
def fix_missing_ffill(df, col):
    count = df[col].isna().sum()
    df[col] = df[col].fillna(method = 'ffill')
    print(f"{count} valores ausentes na coluna {col} foram substituídos usando o método de preenchimento progressivo.")
    return df[col]


# Imputação de valores ausentes usando backward fill
# method = 'bfill': Bfill ou backward-fill propaga o primeiro valor não nulo observado para trás até que outro valor não nulo seja encontrado
def fix_missing_bfill(df, col):
    count = df[col].isna().sum()
    df[col] = df[col].fillna(method = 'bfill')
    print(f"{count} valores ausentes na coluna {col} foram substituídos usando o método de preenchimento reverso.")
    return df[col]


# Imputação usando a mediana
def fix_missing_median(df, col):
    median = df[col].median()
    count = df[col].isna().sum()
    df[col] = df[col].fillna(median)
    print(f"{count} valores ausentes na coluna {col} foram substituídos por seu valor de mediana {median}.")
    return df[col]


# Preenche valor NA
def fix_missing_value(df, col, value):
    count = df[col].isna().sum()
    df[col] = df[col].fillna(value)
    if type(value) == 'str':
        print(f"{count} valores ausentes na coluna {col} foram substituídos por '{value}'.")
    else:
        print(f"{count} valores ausentes na coluna {col} foram substituídos por {value}.")
    return df[col]


# Drop duplicatas
def drop_duplicates(df):
    old = df.shape[0]
    df.drop_duplicates(inplace = True)
    new = df.shape[0]
    count = old - new
    if (count == 0):
        print("Nenhuma linha duplicada foi encontrada.")
    else:
        print(f"{count} linhas duplicadas foram encontradas e removidas.")


# Drop de linhas com valores ausentes
def drop_rows_with_missing_values(df):
    old = df.shape[0]
    df.dropna(inplace = True)
    new = df.shape[0]
    count = old - new
    print(f"{count} linhas contendo valores ausentes foram descartadas.")


# Drop de colunas
def drop_columns(df, columns):
    df.drop(columns, axis = 1, inplace = True)
    count = len(columns)
    if count == 1:
        print(f"{count} coluna foi descartada.")
    else:
        print(f"{count} colunas foram descartadas.")
