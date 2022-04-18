from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import VAR


class AR():
    '''
     Autoregressions (AR)
     y_t = sum{alpha_i*y_i} + res
     y: T * 1
     T: number of time steps

    '''

    def __init__(self, data, p=50):
        '''
        arg data: targets
            p   : max lag order

        '''
        super().__init__()
        self.data = data
        mod = ar_select_order(data, maxlag=p)
        self.results = AutoReg(data, lags=mod.ar_lags).fit()

    def residual(self):
        '''
        transform targets to its AR residuals
        return: AR residuals

        '''
        fit = self.results.fittedvalues
        return self.data - fit

    def predict(self, res):
        '''
        transform predicted residuals to predicted targets
        arg res: predicted residuals, len=predict period
        return: predicted targets
        '''
        n = len(res)
        pred = self.results.forecast(n)
        return pred + res.values


class V_AR():
    '''
    Vector Autoregressions (VAR)
    Y_t = sum{A_i*Y_i} + Res
    Y: T * K
    T: number of time steps
    K: number of features

    '''

    def __init__(self, data, p=20):
        '''
        arg data: features
            p: max lag order

        '''
        self.data = data
        model = VAR(data)
        self.results = model.fit(maxlags=p, ic='aic')

    def residual(self):
        '''
        transform features to its VAR residuals
        return: VAR residuals

        '''
        fit = self.results.fittedvalues
        return self.data - fit
