# Attack Classifier Training Report

**Accuracy:** 0.996117636510356

**F1 Score:** 0.9965095663450719

## Confusion Matrix
[[  392     0     0     1     0     0     0     0     0     0     0     0
      0     0]
 [    3 25600     0     0     0     0     0     0     3     0     0     0
      0     0]
 [    6     0  1956     5     7     2     0    67     3     0     0     8
      4     1]
 [    2     0     1 46135     1     0     0     0     6     2     0    14
     25    29]
 [    0     0     0     2  1083     5     0     0     0     0     0     2
      0     8]
 [    0     0     1    12     7  1129     0     0     0     1     0     6
      0     3]
 [    0     0     0     0     0     0  1569     0    19     0     0     0
      0     0]
 [    0     0     0     1     0     0     0     0     0     0     0     0
      0     1]
 [    0     1     2     0     0     0     0     0     1     0     0     2
      0     1]
 [    0     0     0     6    15     0     0     0     0 31756     0     6
      0     3]
 [    0     0     1     0     0     0     2     0     1     0  1175     0
      1     0]
 [    0     0     0     0     1     0     0     0     0     0     0   291
      0     9]
 [    0     0     0     2     1     0     0     0     0     0     0     1
      0     0]
 [    0     0     0     1     0     0     0     0     0     0     1   118
      0    10]]

## Classification Report
```
                            precision    recall  f1-score   support

                       Bot       0.97      1.00      0.98       393
                      DDoS       1.00      1.00      1.00     25606
             DoS GoldenEye       1.00      0.95      0.97      2059
                  DoS Hulk       1.00      1.00      1.00     46215
          DoS Slowhttptest       0.97      0.98      0.98      1100
             DoS slowloris       0.99      0.97      0.98      1159
               FTP-Patator       1.00      0.99      0.99      1588
                Heartbleed       0.00      0.00      0.00         2
              Infiltration       0.03      0.14      0.05         7
                  PortScan       1.00      1.00      1.00     31786
               SSH-Patator       1.00      1.00      1.00      1180
  Web Attack � Brute Force       0.65      0.97      0.78       301
Web Attack � Sql Injection       0.00      0.00      0.00         4
          Web Attack � XSS       0.15      0.08      0.10       130

                  accuracy                           1.00    111530
                 macro avg       0.70      0.72      0.70    111530
              weighted avg       1.00      1.00      1.00    111530

```
