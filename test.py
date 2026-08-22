import tushare as ts

# 设置token
ts.set_token("bc0a6011b50fe1c60ca4f450725dd86ece270be7a678a7d811b94988")

pro = ts.pro_api()


df = pro.index_daily(
    ts_code="000985.CSI",
    start_date="20110101",
    end_date="20260531"
)

print(df.head())