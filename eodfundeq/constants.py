"""Definitions of important constants."""

from enum import Enum

START_DATE = '1999-12-31'

class TimeSeriesNames(Enum):
    """Enum with names of some important time series."""
    ADJUSTED_CLOSE = 'adjusted_close'
    CLOSE = 'close'
    MARKET_CAP = 'market_cap'
    MONTHLY_RETURNS = 'monthly_returns'
    VOLUME = 'volume'


class ReturnTypes(Enum):
    """An Enum class enumerating the various available return types."""
    ARITHMETIC = 'arith'
    LOG = 'log'


class DataSetTypes(Enum):
    """An Enum class enumerating the various data set types."""
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'

    
class FinancialStatementTypes(Enum):
    """An Enum class containing financial statement types."""
    INCOME_STATEMENT = 'Income_Statement'
    BALANCE_SHEET = 'Balance_Sheet'    
    CASH_FLOW_STATEMENT = 'Cash_Flow'


class FundamentalRatios(Enum):
    """Enum class of the supported fundamental ratios."""
    ROA = 'ROA'
    ROE = 'ROE'
    ROIC = 'ROIC'
    GROSS_MARGIN = 'GrossMargin'
    OPERATING_MARGIN = 'OperatingMargin'
    NET_MARGIN = 'NetMargin'
    CASH_FLOW_MARGIN = 'CashFlowMargin'
    CASH_FLOW_TO_NET_INCOME = 'CashFlowtoNetIncome'
    EARNINGS_YIELD = 'EarningsYield'


class FundamentalRatioInputs(Enum):
    """Enum class of fundamental data used as input for calculating ratios."""
    EBIT = 'ebit'
    FREE_CASH_FLOW = 'freeCashFlow'
    GROSS_PROFIT = 'grossProfit'
    INCOME_BEFORE_TAX = 'incomeBeforeTax'
    INCOME_TAX_EXPENSE = 'incomeTaxExpense'
    NET_INCOME = 'netIncome'
    NET_INVESTED_CAPITAL = 'netInvestedCapital'
    SHAREHOLDER_EQUITY = 'totalStockholderEquity'
    TOTAL_ASSETS = 'totalAssets'    
    TOTAL_REVENUE = 'totalRevenue'


INCOME_STATEMENT_DATA_TYPES = (
     FundamentalRatioInputs.EBIT.value,
     FundamentalRatioInputs.GROSS_PROFIT.value,
     FundamentalRatioInputs.INCOME_BEFORE_TAX.value,
     FundamentalRatioInputs.INCOME_TAX_EXPENSE.value,
     FundamentalRatioInputs.NET_INCOME.value,
     FundamentalRatioInputs.TOTAL_REVENUE.value,
)
        
BALANCE_SHEET_DATA_TYPES = (
    FundamentalRatioInputs.NET_INVESTED_CAPITAL.value,
    FundamentalRatioInputs.TOTAL_ASSETS.value,
    FundamentalRatioInputs.SHAREHOLDER_EQUITY.value,
)

CASH_FLOW_STATEMENT_DATA_TYPES = (
    FundamentalRatioInputs.FREE_CASH_FLOW.value,
)

FINANCIAL_DATA_TYPE_MAP = {
    FinancialStatementTypes.BALANCE_SHEET.value: BALANCE_SHEET_DATA_TYPES,
    FinancialStatementTypes.CASH_FLOW_STATEMENT.value: CASH_FLOW_STATEMENT_DATA_TYPES,
    FinancialStatementTypes.INCOME_STATEMENT.value: INCOME_STATEMENT_DATA_TYPES,
}