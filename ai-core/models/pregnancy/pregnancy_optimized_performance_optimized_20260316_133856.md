# تقرير أداء نموذج الحمل المحسن (Optimized for Mid Risk)

## 🎯 النتائج النهائية
- **الإصدار**: optimized_20260316_133856
- **النموذج**: Ensemble مع GridSearch
- **تاريخ التدريب**: 2026-03-16 13:38
- **Macro F1-Score**: 0.8558

## 📈 أداء كل فئة
| الفئة | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| low risk | 0.859 | 0.827 | 0.843 |
| mid risk | 0.815 | 0.791 | 0.803 |
| high risk | 0.883 | 0.964 | 0.922 |

## 🎯 تحسين Mid Risk
- **F1-Score**: 0.8030
- **Recall**: 0.7910
- **التحسن**: هدفنا الوصول بـ Recall > 0.80

## 📊 أهم 10 ميزات
14. **BP_BS_Interaction**: 0.1710
4. **BS**: 0.1493
16. **Age_BP_Interaction**: 0.0797
1. **Age**: 0.0723
11. **MeanBP**: 0.0673
18. **BS_Severity**: 0.0645
2. **SystolicBP**: 0.0629
5. **BodyTemp**: 0.0578
6. **HeartRate**: 0.0575
8. **BP_ratio**: 0.0535

## 📊 Cross Validation
- **3-Folds F1 Macro**: 0.8037 ± 0.0571

## 🔍 التوصيات
1. ✅ Mid Risk أداء ممتاز
2. ✅ Recall جيد
