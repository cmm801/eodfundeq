"""Definitions of important constants."""

from enum import Enum
from eodhistdata.constants import FundamentalDataTypes


START_DATE = '1999-12-31'


class ModelTypes(Enum):
    BULL = 'bull'
    BEAR = 'bear'


class ReturnTypes(Enum):
    """An Enum class enumerating the various available return types."""
    ARITHMETIC = 'arith'
    LOG = 'log'


class DataSetTypes(Enum):
    """An Enum class enumerating the various data set types."""
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


class FundamentalRatios(Enum):
    """Enum class of the supported fundamental ratios."""
    ASSETS_GROWTH = 'AssetGrowth'    
    BOOK_TO_PRICE = 'BookToPrice'
    BOOK_VALUE_GROWTH = 'BookValueGrowth'
    CAPEX_GROWTH = 'CapExGrowth'
    CASH_FLOW_MARGIN = 'CashFlowMargin'
    CASH_FLOW_TO_NET_INCOME = 'CashFlowtoNetIncome'
    EARNINGS_GROWTH = 'EarningsGrowth'
    EARNINGS_TO_PRICE = 'EarningsToPrice'    
    EBITDA_TO_EV = 'EBITDAToEV'
    EQUITY_ISSUANCE = 'EquityIssuance'
    FIXED_ASSETS_GROWTH = 'FixedAssetsGrowth'
    FREE_CASH_FLOW_TO_EV = 'FreeCashFlowToEV'
    FREE_CASH_FLOW_YIELD = 'FreeCashFlowYield'
    GROSS_MARGIN = 'GrossMargin'
    NET_DEBT_GROWTH = 'NetDebtGrowth'
    NET_MARGIN = 'NetMargin'
    NET_PAYOUT_YIELD = 'NetPayoutYield'
    OPERATING_MARGIN = 'OperatingMargin'    
    ROA = 'ROA'
    ROE = 'ROE'
    ROIC = 'ROIC'
    SALES_TO_EV = 'SalesToEV'
    SALES_TO_PRICE = 'SalesToPrice'

# Fundamental ratios that measure valuation of a stock
VALUATION_RATIOS = (
    FundamentalRatios.BOOK_TO_PRICE.value,
    FundamentalRatios.EARNINGS_TO_PRICE.value,
    FundamentalRatios.EBITDA_TO_EV.value,
    FundamentalRatios.FREE_CASH_FLOW_TO_EV.value,
    FundamentalRatios.SALES_TO_EV.value,
    FundamentalRatios.SALES_TO_PRICE.value,
)

# Fundamental ratios that measure profitability] of a company
PROFITABILITY_RATIOS = (
    FundamentalRatios.CASH_FLOW_MARGIN.value,
    FundamentalRatios.GROSS_MARGIN.value,
    FundamentalRatios.OPERATING_MARGIN.value,    
    FundamentalRatios.ROA.value,
    FundamentalRatios.ROE.value,
    FundamentalRatios.ROIC.value,
)

PAYOUT_DILUTION_RATIOS = (
    FundamentalRatios.EQUITY_ISSUANCE.value,
    FundamentalRatios.NET_DEBT_GROWTH.value,
    FundamentalRatios.NET_PAYOUT_YIELD.value,
)

"""Tuple of fundamental data used as input for calculating fundamental ratios."""
FUNDAMENTAL_RATIO_INPUTS = (
    FundamentalDataTypes.capitalExpenditures.value,
    FundamentalDataTypes.cashAndEquivalents.value,
    FundamentalDataTypes.commonStockSharesOutstanding.value,
    FundamentalDataTypes.dividendsPaid.value,
    FundamentalDataTypes.ebit.value,
    FundamentalDataTypes.ebitda.value,
    FundamentalDataTypes.dividendsPaid.value,
    FundamentalDataTypes.freeCashFlow.value,
    FundamentalDataTypes.grossProfit.value,
    FundamentalDataTypes.incomeBeforeTax.value,
    FundamentalDataTypes.incomeTaxExpense.value,
    FundamentalDataTypes.longTermDebt.value,
    FundamentalDataTypes.netDebt.value,
    FundamentalDataTypes.netIncome.value,
    FundamentalDataTypes.netInvestedCapital.value,
    FundamentalDataTypes.propertyPlantEquipment.value,
    FundamentalDataTypes.salePurchaseOfStock.value,
    FundamentalDataTypes.shortTermDebt.value,
    FundamentalDataTypes.totalStockholderEquity.value,
    FundamentalDataTypes.totalAssets.value,
    FundamentalDataTypes.totalRevenue.value,
)