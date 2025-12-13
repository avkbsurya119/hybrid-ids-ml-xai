# Attack Classifier Training Report

**Accuracy:** 0.9959966456411565

**F1 Score:** 0.9965021283238161

## Confusion Matrix
[[  386     0     0     0     0     0     0     0     0     0     2     0
      0     5]
 [    2 51146     0     0     0     0     0     0    18     0     0    27
      0    18]
 [    3     0  2015    21     3     2     0     0     2     0     1     7
      2     3]
 [    1     2    17 46062     2     0     0     1    80     1     0    31
     13     5]
 [    0     0     1     0  1084     8     0     0     0     0     0     4
      0     3]
 [    1     0     1     4     6  1131     6     0     0     1     1     4
      0     4]
 [    0     0     0     0     0     2  1580     0     2     0     2     2
      0     0]
 [    0     0     1     0     0     0     0     0     0     0     0     0
      0     1]
 [    0     2     1     1     1     0     0     0     0     0     2     0
      0     0]
 [    1     0     0    11     1     0     2     0     0 31764     2     2
      1     2]
 [    0     0     1     0     0     0     7     0     0     0  1170     0
      1     1]
 [    0     5     8    16     5     1     3     0    10     8     0   192
      0    53]
 [    1     0     0     1     0     1     0     0     0     0     0     1
      0     0]
 [    0     3     1     7     4     1     2     0     5     8     0    43
      0    56]]

## Classification Report
```
                            precision    recall  f1-score   support

                       Bot       0.98      0.98      0.98       393
                      DDoS       1.00      1.00      1.00     51211
             DoS GoldenEye       0.98      0.98      0.98      2059
                  DoS Hulk       1.00      1.00      1.00     46215
          DoS Slowhttptest       0.98      0.99      0.98      1100
             DoS slowloris       0.99      0.98      0.98      1159
               FTP-Patator       0.99      0.99      0.99      1588
                Heartbleed       0.00      0.00      0.00         2
              Infiltration       0.00      0.00      0.00         7
                  PortScan       1.00      1.00      1.00     31786
               SSH-Patator       0.99      0.99      0.99      1180
  Web Attack � Brute Force       0.61      0.64      0.63       301
Web Attack � Sql Injection       0.00      0.00      0.00         4
          Web Attack � XSS       0.37      0.43      0.40       130

                  accuracy                           1.00    137135
                 macro avg       0.71      0.71      0.71    137135
              weighted avg       1.00      1.00      1.00    137135

```
