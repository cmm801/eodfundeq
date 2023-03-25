"""Definitions of important constants."""

from enum import Enum
from eodhistdata.constants import FundamentalDataTypes


START_DATE = '1999-12-31'


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