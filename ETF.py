import numpy as np
import pandas as pd
import yfinance as yf
import openpyxl
from yahoofinancials import YahooFinancials

START_DATE = '2016-01-01'
END_DATE = '2021-01-01'
FREQUENCY = 'daily'


class ETF:
    """
    A class used to retrieve ETF data from Yahoo Finance API

    Attributes:
    -----------
    start_date : str
        Start date of the data retrieval in 'YYYY-MM-DD' format. Default is '2016-01-01'
    end_date : str
        End date of the data retrieval in 'YYYY-MM-DD' format. Default is '2021-01-01'
    frequency : str
        Frequency of the data to be retrieved. Default is 'daily'

    Methods:
    --------
    get_etf_data()
        Retrieves ETF data from Yahoo Finance API for the specified date range and frequency. Returns a numpy array.
    """

    def __init__(self, start_date=START_DATE, end_date=END_DATE, frequency=FREQUENCY):
        self.etfs = None
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency

    def get_etf_data(self):
        """
        Retrieves ETF data from Yahoo Finance API for the specified date range and frequency.

        Returns:
        --------
        numpy.ndarray
            A numpy array of ETF prices for each ticker
        """
        prices = []
        etf_list = pd.read_excel('ETF_Detail.xlsx', header=None, dtype={'ETFs': str, 'Value': str})
        self.etfs = np.array(etf_list[0])

        for etf in self.etfs:
            yahoo_financials = YahooFinancials(etf)
            data = yahoo_financials.get_historical_price_data(start_date=self.start_date,
                                                              end_date=self.end_date,
                                                              time_interval=self.frequency)
            data = pd.DataFrame(data[etf]['prices'])
            data = np.array(data['adjclose'])
            prices.append(data)

        prices = np.array(prices)
        return prices


if __name__ == '__main__':

    etf_obj = ETF()
    etf_prices = etf_obj.get_etf_data()
    print(etf_prices)
