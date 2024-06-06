from alpaca.data.timeframe import TimeFrame



symbol_map={'ADABNB': 0,
 'ADABTC': 1,
 'ADAETH': 2,
 'ADAUSDT': 3,
 'ADAUSD': 3,
 'BNBETH': 4,
 'BNBUSDT': 5,
 'BNBUSD': 5,
 'BTCUSDT': 6,
 'BTCUSD': 6,
 'ETHBTC': 7,
 'ETHUSDT': 8,
 'ETHUSD': 8,
 'SOLBNB': 9,
 'SOLBTC': 10,
 'SOLETH': 11,
 'SOLUSDT': 12,
 'SOLUSD': 12,
 'XRPBNB': 13,
 'XRPBTC': 14,
 'XRPETH': 15,
 'XRPUSDT': 16,
 'XRPUSD': 16,
 }



binanace_col_map = {
                    "t": "date_open",
                    "T": "date_close",
                    "s": "symbol",
                    "o": "open",
                    "c": "close",
                    "h": "high",
                    "l": "low",
                    "v": "volume",
                    "x": "is_closed",
                }
alpaca_stream_col_map=binanace_col_map
alpaca_stream_col_map.update({
                        't':'date_close',
                        'T':"bar_type",
                        "S":'symbol'
                       }
                       )
alpaca_stream_message_map={
                        'b':'Bar',
                        'd':'Daily Bar',
                        'u':"Updated Bar"
                    }
alpaca_time_map={
    '1h':TimeFrame.Hour,
    '1d':TimeFrame.Day,
}