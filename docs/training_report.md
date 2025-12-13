# Training Report (LightGBM)

**Accuracy:** 0.9982561696586607

**F1 Score:** 0.9983590407789515

## Confusion Matrix
[[473297    295     12     47    123    112     12      5      0     15
     189     11     22      5     18]
 [     2    391      0      0      0      0      0      0      0      0
       0      0      0      0      0]
 [     3      0  51208      0      0      0      0      0      0      0
       0      0      0      0      0]
 [     0      0      0   2059      0      0      0      0      0      0
       0      0      0      0      0]
 [     1      0      0     15  46197      0      0      0      0      0
       1      0      0      1      0]
 [     1      0      0      1      0   1096      2      0      0      0
       0      0      0      0      0]
 [     1      0      0      0      0      5   1149      0      0      0
       0      0      3      1      0]
 [     0      0      0      0      0      0      0   1587      0      0
       0      0      1      0      0]
 [     0      0      0      0      0      0      0      0      2      0
       0      0      0      0      0]
 [     1      0      0      0      0      0      0      0      0      6
       0      0      0      0      0]
 [     3      0      0      0      8      0      1      0      0      0
   31773      0      0      0      1]
 [     0      0      0      0      0      0      0      0      0      0
       0   1180      0      0      0]
 [     0      0      0      0      0      0      0      0      0      0
       0      0    203      0     98]
 [     0      0      0      2      0      0      0      0      0      0
       0      0      0      2      0]
 [     3      0      0      0      0      0      0      0      0      0
       0      0     45      0     82]]

## Classification Report
```
                            precision    recall  f1-score   support

                    BENIGN       1.00      1.00      1.00    474163
                       Bot       0.57      0.99      0.72       393
                      DDoS       1.00      1.00      1.00     51211
             DoS GoldenEye       0.97      1.00      0.98      2059
                  DoS Hulk       1.00      1.00      1.00     46215
          DoS Slowhttptest       0.90      1.00      0.95      1100
             DoS slowloris       0.99      0.99      0.99      1159
               FTP-Patator       1.00      1.00      1.00      1588
                Heartbleed       1.00      1.00      1.00         2
              Infiltration       0.29      0.86      0.43         7
                  PortScan       0.99      1.00      1.00     31786
               SSH-Patator       0.99      1.00      1.00      1180
  Web Attack � Brute Force       0.74      0.67      0.71       301
Web Attack � Sql Injection       0.22      0.50      0.31         4
          Web Attack � XSS       0.41      0.63      0.50       130

                  accuracy                           1.00    611298
                 macro avg       0.80      0.91      0.84    611298
              weighted avg       1.00      1.00      1.00    611298

```
