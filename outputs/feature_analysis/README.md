# 特征重要性分析报告

## 分析概述
- **分析时间**: 2025-09-06 15:50:04
- **数据规模**: 260,864 样本 × 62 特征
- **正样本比例**: 6.12%

## 分析方法
✅ 相关性分析 (Pearson & Spearman)
✅ 统计检验 (卡方检验 & F检验)
✅ 模型重要性 (Random Forest, LightGBM, XGBoost)
✅ 排列重要性分析
✅ 特征稳定性分析
✅ 综合特征排名

## 关键发现

### TOP 10 重要特征
1. **user_repeat_count** (得分: 0.8088)
2. **user_repeat_rate** (得分: 0.5197)
3. **loyalty_vs_merchant_rate** (得分: 0.2414)
4. **merchant_repeat_std** (得分: 0.2036)
5. **user_loyalty_score** (得分: 0.1855)
6. **loyalty_x_large_merchant** (得分: 0.1693)
7. **merchant_repeat_stability** (得分: 0.1550)
8. **avg_actions_per_time** (得分: 0.1057)
9. **item_diversity** (得分: 0.1002)
10. **click_to_buy_rate** (得分: 0.0926)

## 文件说明
- `plots/`: 所有分析图表
- `feature_importance_report.json`: 完整分析结果
- `feature_recommendation_top30.json`: 特征选择建议

## 建议
基于分析结果，建议在模型训练中优先使用TOP 30特征，可以在保持性能的同时降低模型复杂度。

---
生成时间: 2025-09-06 15:50:04
