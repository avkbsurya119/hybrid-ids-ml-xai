# Feature Report Summary

Generated from 24 parquet parts. Rows sampled: 200000.

## Top 10 features by missing percentage
| feature                     |   missing_pct | dtype   |
|:----------------------------|--------------:|:--------|
| Destination Port            |             0 | int64   |
| Flow Duration               |             0 | int64   |
| Total Fwd Packets           |             0 | int64   |
| Total Backward Packets      |             0 | int64   |
| Total Length of Fwd Packets |             0 | int64   |
| Total Length of Bwd Packets |             0 | int64   |
| Fwd Packet Length Max       |             0 | int64   |
| Fwd Packet Length Min       |             0 | int64   |
| Fwd Packet Length Mean      |             0 | float64 |
| Fwd Packet Length Std       |             0 | float64 |

## Numeric feature summary (top skewed)
| feature                     |     skew |   kurtosis |         mean |              std |             p99 |
|:----------------------------|---------:|-----------:|-------------:|-----------------:|----------------:|
| ECE Flag Count              | 158.106  |    24995.6 |      4e-05   |      0.00632444  |     0           |
| RST Flag Count              | 158.106  |    24995.6 |      4e-05   |      0.00632444  |     0           |
| Flow IAT Min                | 123.644  |    16478.6 |   8904.67    | 467049           | 82643.8         |
| Active Std                  | 108.526  |    20547.6 |   8793.57    | 186547           | 95234           |
| act_data_pkt_fwd            |  88.765  |    12712.4 |      3.19185 |      9.55855     |    17           |
| Flow Bytes/s                |  85.9663 |     9082.3 | 442484       |      1.51823e+07 |     2.34147e+06 |
| Total Length of Bwd Packets |  84.5335 |    10544   |   5926.23    |  31391.6         | 11607           |
| Subflow Bwd Bytes           |  84.5335 |    10544   |   5926.23    |  31391.6         | 11607           |
| Subflow Bwd Packets         |  75.5724 |     9230   |      4.31457 |     16.923       |    21           |
| Total Backward Packets      |  75.5724 |     9230   |      4.31457 |     16.923       |    21           |
