import pandas as pd

# git clone https://github.com/manu-mannattil/nolitsa.git
# pip install ./nolitsa
from nolitsa import surrogates


def iaaft(x=None):
    """ Function for the surrogates generation"""
    y, i, e = surrogates.iaaft(x)
    return y.tolist()


def preprocess_dataframe(data):
    """Helper method to preprocess the dataframe.
       Creates new columns for year,month,recalls and percentage change.
       Limits the date range for the experiment (these data are trustworthy)."""
    data['recalls'] = data['doc_count'] + 1
    data.drop(columns=['product', 'Unnamed: 0', 'key', 'key_as_string', 'doc_count'], inplace=True)
    data = data.resample("M").sum()
    mask = (data.index > '2007-05-31') & (data.index < '2019-09-30')
    data = data.loc[mask]
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['pct'] = data['recalls'].pct_change()
    return data


def create_surrogate(data, factor, n_samples):
    """ Given the data and the factor of surrogation required, this method
        generates new data and merges them with the existing ones."""
    df_surrogate = data.copy()[0:n_samples * factor]
    percentages_data = df_surrogate.pct.to_numpy()
    surrogate_data = iaaft(percentages_data[:n_samples * factor])
    df_surrogate.pct = surrogate_data
    df_surrogate = df_surrogate.append(data)
    data = df_surrogate
    data.dropna(inplace=True)
    data.sort_index(inplace=True)
    return data





def write_results(results, factor, test_num):
    """ Helper method used to save the result of each experiment in the corresponding file."""
    with pd.ExcelWriter('./results/gluon_ts_surrogation_' + str(factor) + '_test_' + str(test_num) + '.xlsx') as writer:
        results[0][1].to_excel(writer, sheet_name='Deep ARE Estimator')
        results[1][1].to_excel(writer, sheet_name='Simple FF Estimator')
        results[2][1].to_excel(writer, sheet_name='Deep Factor Estimator')
        results[2][1].to_excel(writer, sheet_name='Seasonal Estimator')
        results[4][1].to_excel(writer, sheet_name='MQCNN Estimator')
        results[3][1].to_excel(writer, sheet_name='WaveNet Estimator')
