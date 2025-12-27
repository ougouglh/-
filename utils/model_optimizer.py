import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import pickle
import json

# 机器学习库
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb

# 采样技术
from sklearn.utils import resample

warnings.filterwarnings('ignore')


class ModelOptimizerDDrive:
    """模型性能优化器 - D盘版本"""

    def __init__(self, feature_timestamp='20250910_051339', base_dir='D:/repeat_buyer_prediction'):
        """
        初始化优化器 - 使用D盘路径

        Args:
            feature_timestamp: 使用的特征时间戳
            base_dir: 项目基础目录（D盘）
        """
        self.feature_timestamp = feature_timestamp
        self.base_dir = base_dir

        # 设置路径
        self.feature_dir = os.path.join(base_dir, 'outputs', 'features_time_aware')
        self.optimization_dir = os.path.join(base_dir, 'outputs', 'model_optimization')
        self.data_dir = os.path.join(base_dir, 'data', 'data_format1')

        os.makedirs(self.optimization_dir, exist_ok=True)

        print(f"模型优化器初始化完成 (D盘版本)")
        print(f"基础目录: {self.base_dir}")
        print(f"特征目录: {self.feature_dir}")
        print(f"目标: 在保持时间安全的前提下提升AUC性能")

    def load_data(self):
        """加载时间感知特征数据"""
        print("加载时间感知特征数据...")

        train_path = os.path.join(self.feature_dir, f'train_features_time_aware_{self.feature_timestamp}.csv')
        test_path = os.path.join(self.feature_dir, f'test_features_time_aware_{self.feature_timestamp}.csv')

        # 检查文件是否存在
        if not os.path.exists(train_path):
            print(f"错误: 训练特征文件不存在: {train_path}")
            print("请确保已将特征文件复制到D盘对应目录")
            raise FileNotFoundError(f"特征文件不存在: {train_path}")

        self.train_features = pd.read_csv(train_path)
        self.test_features = pd.read_csv(test_path)

        # 准备训练数据
        feature_cols = [col for col in self.train_features.columns
                        if col not in ['user_id', 'merchant_id', 'label']]

        self.X_train = self.train_features[feature_cols]
        self.y_train = self.train_features['label']
        self.X_test = self.test_features[feature_cols]
        self.feature_names = feature_cols

        # 清理数据
        self.X_train = self.X_train.fillna(0).replace([np.inf, -np.inf], 0)
        self.X_test = self.X_test.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"数据加载完成: 训练集{self.X_train.shape}, 特征数{len(self.feature_names)}")

    def create_advanced_features(self):
        """创建高级特征（保持时间安全）"""
        print("\n=== 创建高级特征 ===")

        X_train_enhanced = self.X_train.copy()
        X_test_enhanced = self.X_test.copy()

        # 1. 特征交叉
        print("创建特征交叉...")
        cross_features = [
            # 用户行为相关交叉
            ('user_hist_purchase_rate', 'user_hist_total_actions'),
            ('user_hist_loyalty_score', 'user_hist_unique_merchants'),
            ('user_hist_merchant_diversity', 'user_hist_avg_daily_actions'),

            # 商家特征交叉
            ('merchant_hist_repeat_user_rate', 'merchant_hist_unique_users'),
            ('merchant_hist_purchase_rate', 'merchant_hist_avg_user_actions'),
            ('merchant_hist_user_loyalty_score', 'merchant_hist_popularity_score'),

            # 用户-商家交互交叉
            ('interaction_hist_purchase_rate', 'interaction_hist_total_actions'),
            ('interaction_hist_frequency_score', 'has_historical_interaction'),
        ]

        for f1, f2 in cross_features:
            if f1 in X_train_enhanced.columns and f2 in X_train_enhanced.columns:
                # 乘积特征
                X_train_enhanced[f'{f1}_x_{f2}'] = X_train_enhanced[f1] * X_train_enhanced[f2]
                X_test_enhanced[f'{f1}_x_{f2}'] = X_test_enhanced[f1] * X_test_enhanced[f2]

                # 比值特征（避免除零）
                X_train_enhanced[f'{f1}_div_{f2}'] = X_train_enhanced[f1] / (X_train_enhanced[f2] + 1e-8)
                X_test_enhanced[f'{f1}_div_{f2}'] = X_test_enhanced[f1] / (X_test_enhanced[f2] + 1e-8)

        # 2. 统计特征
        print("创建统计特征...")

        # 用户相关统计
        user_features = [col for col in X_train_enhanced.columns if col.startswith('user_hist_')]
        if len(user_features) > 3:
            X_train_enhanced['user_features_mean'] = X_train_enhanced[user_features].mean(axis=1)
            X_train_enhanced['user_features_std'] = X_train_enhanced[user_features].std(axis=1)
            X_train_enhanced['user_features_max'] = X_train_enhanced[user_features].max(axis=1)

            X_test_enhanced['user_features_mean'] = X_test_enhanced[user_features].mean(axis=1)
            X_test_enhanced['user_features_std'] = X_test_enhanced[user_features].std(axis=1)
            X_test_enhanced['user_features_max'] = X_test_enhanced[user_features].max(axis=1)

        # 商家相关统计
        merchant_features = [col for col in X_train_enhanced.columns if col.startswith('merchant_hist_')]
        if len(merchant_features) > 3:
            X_train_enhanced['merchant_features_mean'] = X_train_enhanced[merchant_features].mean(axis=1)
            X_train_enhanced['merchant_features_std'] = X_train_enhanced[merchant_features].std(axis=1)

            X_test_enhanced['merchant_features_mean'] = X_test_enhanced[merchant_features].mean(axis=1)
            X_test_enhanced['merchant_features_std'] = X_test_enhanced[merchant_features].std(axis=1)

        # 3. 排序特征
        print("创建排序特征...")

        # 用户在所有用户中的排名特征
        for col in ['user_hist_total_actions', 'user_hist_purchase_count', 'user_hist_unique_merchants']:
            if col in X_train_enhanced.columns:
                X_train_enhanced[f'{col}_rank'] = X_train_enhanced[col].rank(pct=True)
                # 对测试集，使用训练集的分位数
                train_quantiles = X_train_enhanced[col].quantile(np.linspace(0, 1, 100))
                X_test_enhanced[f'{col}_rank'] = X_test_enhanced[col].apply(
                    lambda x: np.searchsorted(train_quantiles.values, x) / 100
                )

        # 清理新特征
        X_train_enhanced = X_train_enhanced.fillna(0).replace([np.inf, -np.inf], 0)
        X_test_enhanced = X_test_enhanced.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"高级特征创建完成: 从{self.X_train.shape[1]}个特征扩展到{X_train_enhanced.shape[1]}个")

        return X_train_enhanced, X_test_enhanced

    def feature_selection_optimization(self, X_train, X_test, y_train):
        """优化特征选择"""
        print("\n=== 特征选择优化 ===")

        # 1. 统计特征选择
        print("1. 统计特征选择...")
        selector_f = SelectKBest(f_classif, k=min(80, X_train.shape[1]))
        X_train_selected = selector_f.fit_transform(X_train, y_train)
        X_test_selected = selector_f.transform(X_test)

        selected_features = X_train.columns[selector_f.get_support()]
        print(f"   F检验选择了{len(selected_features)}个特征")

        # 2. 基于模型的特征选择
        print("2. 基于模型的特征选择...")
        lgb_selector = lgb.LGBMClassifier(
            objective='binary',
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_estimators=50
        )

        rfe_selector = RFE(
            estimator=lgb_selector,
            n_features_to_select=min(60, X_train.shape[1]),
            step=5
        )

        X_train_rfe = rfe_selector.fit_transform(X_train, y_train)
        X_test_rfe = rfe_selector.transform(X_test)

        rfe_features = X_train.columns[rfe_selector.support_]
        print(f"   RFE选择了{len(rfe_features)}个特征")

        # 3. 组合特征选择（取交集）
        common_features = list(set(selected_features) & set(rfe_features))
        print(f"   最终选择{len(common_features)}个共同重要特征")

        if len(common_features) < 20:  # 如果交集太少，取并集
            common_features = list(set(selected_features) | set(rfe_features))
            print(f"   交集过少，使用并集: {len(common_features)}个特征")

        X_train_final = X_train[common_features]
        X_test_final = X_test[common_features]

        return X_train_final, X_test_final, common_features

    def advanced_sampling_strategies(self, X_train, y_train):
        """高级采样策略"""
        print("\n=== 高级采样策略 ===")

        sampling_strategies = {}

        # 1. 原始数据
        sampling_strategies['original'] = (X_train, y_train)

        # 2. 智能欠采样 - 移除边界附近的负样本
        print("1. 智能欠采样...")

        # 训练一个简单模型识别难分样本
        simple_model = lgb.LGBMClassifier(
            objective='binary',
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_estimators=50
        )
        simple_model.fit(X_train, y_train)

        # 预测概率
        pred_proba = simple_model.predict_proba(X_train)[:, 1]

        # 保留所有正样本
        pos_indices = y_train[y_train == 1].index

        # 对负样本进行智能采样：保留预测概率较高的负样本（困难样本）
        neg_indices = y_train[y_train == 0].index
        neg_proba = pred_proba[neg_indices]

        # 选择概率较高的负样本和一些随机负样本
        hard_neg_count = len(pos_indices) * 3  # 3:1比例
        easy_neg_count = len(pos_indices) * 1  # 额外的简单负样本

        hard_neg_indices = neg_indices[np.argsort(neg_proba)[-hard_neg_count:]]
        easy_neg_indices = np.random.choice(
            neg_indices[np.argsort(neg_proba)[:-hard_neg_count]],
            size=min(easy_neg_count, len(neg_indices) - hard_neg_count),
            replace=False
        )

        selected_indices = np.concatenate([pos_indices, hard_neg_indices, easy_neg_indices])
        X_smart = X_train.loc[selected_indices]
        y_smart = y_train.loc[selected_indices]

        sampling_strategies['smart_undersample'] = (X_smart, y_smart)
        print(f"   智能欠采样: {X_smart.shape[0]}样本, 正样本率: {y_smart.mean() * 100:.2f}%")

        # 3. 边界SMOTE-like - 手工实现简单版本
        print("2. 边界过采样...")

        # 找到正样本中的边界样本（预测概率较低的）
        pos_proba = pred_proba[pos_indices]
        boundary_pos_indices = pos_indices[np.argsort(pos_proba)[:len(pos_indices) // 2]]

        # 对边界正样本进行简单的噪声增强
        boundary_samples = X_train.loc[boundary_pos_indices]

        # 添加高斯噪声生成新样本
        noise_scale = 0.1
        augmented_samples = []
        augmented_labels = []

        for _ in range(len(pos_indices)):  # 生成与正样本数量相等的增强样本
            base_sample = boundary_samples.iloc[np.random.randint(len(boundary_samples))]
            noise = np.random.normal(0, noise_scale, len(base_sample))

            # 只对数值特征添加噪声
            augmented_sample = base_sample.copy()
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in augmented_sample.index:
                    augmented_sample[col] += noise[list(augmented_sample.index).index(col)]

            augmented_samples.append(augmented_sample)
            augmented_labels.append(1)

        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            augmented_df.index = range(len(X_train), len(X_train) + len(augmented_df))

            X_augmented = pd.concat([X_train, augmented_df])
            y_augmented = pd.concat([y_train, pd.Series(augmented_labels, index=augmented_df.index)])

            sampling_strategies['boundary_augmentation'] = (X_augmented, y_augmented)
            print(f"   边界增强: {X_augmented.shape[0]}样本, 正样本率: {y_augmented.mean() * 100:.2f}%")

        return sampling_strategies

    def hyperparameter_optimization(self, X_train, y_train):
        """超参数优化 - 现在应该不会有路径问题"""
        print("\n=== 超参数优化 ===")

        # LightGBM超参数搜索空间
        lgb_param_dist = {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.05, 0.1, 0.15],
            'num_leaves': [31, 50, 80],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }

        # XGBoost超参数搜索空间
        xgb_param_dist = {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 6, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }

        best_models = {}

        # 优化LightGBM
        print("1. 优化LightGBM...")
        lgb_base = lgb.LGBMClassifier(
            objective='binary',
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )

        lgb_search = RandomizedSearchCV(
            lgb_base,
            lgb_param_dist,
            n_iter=20,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1  # D盘路径应该不会有编码问题了
        )

        lgb_search.fit(X_train, y_train)
        best_models['lightgbm'] = lgb_search.best_estimator_
        print(f"   最佳LightGBM AUC: {lgb_search.best_score_:.4f}")

        # 优化XGBoost
        print("2. 优化XGBoost...")
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        xgb_base = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=pos_weight,
            random_state=42
        )

        xgb_search = RandomizedSearchCV(
            xgb_base,
            xgb_param_dist,
            n_iter=20,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1
        )

        xgb_search.fit(X_train, y_train)
        best_models['xgboost'] = xgb_search.best_estimator_
        print(f"   最佳XGBoost AUC: {xgb_search.best_score_:.4f}")

        return best_models

    def ensemble_modeling(self, best_models, X_train, y_train, X_test):
        """集成建模"""
        print("\n=== 集成建模 ===")

        # 1. 简单投票集成
        print("1. 构建投票集成...")
        voting_clf = VotingClassifier(
            estimators=[
                ('lgb', best_models['lightgbm']),
                ('xgb', best_models['xgboost'])
            ],
            voting='soft'  # 使用概率投票
        )

        # 交叉验证评估集成效果
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ensemble_scores = []
        lgb_scores = []
        xgb_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # 训练集成模型
            voting_clf.fit(X_fold_train, y_fold_train)
            ensemble_pred = voting_clf.predict_proba(X_fold_val)[:, 1]
            ensemble_scores.append(roc_auc_score(y_fold_val, ensemble_pred))

            # 单独评估各模型
            best_models['lightgbm'].fit(X_fold_train, y_fold_train)
            lgb_pred = best_models['lightgbm'].predict_proba(X_fold_val)[:, 1]
            lgb_scores.append(roc_auc_score(y_fold_val, lgb_pred))

            best_models['xgboost'].fit(X_fold_train, y_fold_train)
            xgb_pred = best_models['xgboost'].predict_proba(X_fold_val)[:, 1]
            xgb_scores.append(roc_auc_score(y_fold_val, xgb_pred))

        print(f"   LightGBM平均AUC: {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}")
        print(f"   XGBoost平均AUC: {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}")
        print(f"   集成模型平均AUC: {np.mean(ensemble_scores):.4f} ± {np.std(ensemble_scores):.4f}")

        # 训练最终集成模型
        voting_clf.fit(X_train, y_train)

        # 生成预测
        ensemble_pred = voting_clf.predict_proba(X_test)[:, 1]

        return voting_clf, {
            'ensemble_auc': np.mean(ensemble_scores),
            'ensemble_std': np.std(ensemble_scores),
            'lgb_auc': np.mean(lgb_scores),
            'xgb_auc': np.mean(xgb_scores)
        }, ensemble_pred

    def run_optimization_pipeline(self):
        """运行完整的优化流程"""
        print("开始模型优化流程...")

        # 1. 加载数据
        self.load_data()

        # 2. 创建高级特征
        X_train_enhanced, X_test_enhanced = self.create_advanced_features()

        # 3. 特征选择优化
        X_train_selected, X_test_selected, selected_features = self.feature_selection_optimization(
            X_train_enhanced, X_test_enhanced, self.y_train
        )

        # 4. 高级采样策略
        sampling_strategies = self.advanced_sampling_strategies(X_train_selected, self.y_train)

        # 5. 为每种采样策略寻找最佳模型
        best_strategy = None
        best_performance = 0
        best_models_dict = {}

        for strategy_name, (X_strategy, y_strategy) in sampling_strategies.items():
            print(f"\n测试采样策略: {strategy_name}")

            # 超参数优化
            strategy_models = self.hyperparameter_optimization(X_strategy, y_strategy)

            # 快速评估性能
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in cv.split(X_strategy, y_strategy):
                X_fold_train = X_strategy.iloc[train_idx]
                X_fold_val = X_strategy.iloc[val_idx]
                y_fold_train = y_strategy.iloc[train_idx]
                y_fold_val = y_strategy.iloc[val_idx]

                strategy_models['lightgbm'].fit(X_fold_train, y_fold_train)
                pred = strategy_models['lightgbm'].predict_proba(X_fold_val)[:, 1]
                cv_scores.append(roc_auc_score(y_fold_val, pred))

            strategy_performance = np.mean(cv_scores)
            print(f"   策略{strategy_name}性能: {strategy_performance:.4f}")

            if strategy_performance > best_performance:
                best_performance = strategy_performance
                best_strategy = strategy_name
                best_models_dict = strategy_models
                best_X_train, best_y_train = X_strategy, y_strategy

        print(f"\n最佳采样策略: {best_strategy} (AUC: {best_performance:.4f})")

        # 6. 使用最佳策略进行集成建模
        ensemble_model, ensemble_results, final_predictions = self.ensemble_modeling(
            best_models_dict, best_X_train, best_y_train, X_test_selected
        )

        # 7. 保存优化结果
        self._save_optimization_results(
            ensemble_model, ensemble_results, final_predictions,
            selected_features, best_strategy
        )

        print(f"\n=== 优化完成 ===")
        print(f"原始基线AUC: ~0.6667")
        print(f"优化后AUC: {ensemble_results['ensemble_auc']:.4f}")
        print(f"性能提升: {(ensemble_results['ensemble_auc'] - 0.6667) * 100:.2f} 百分点")

        return ensemble_model, ensemble_results, final_predictions

    def _save_optimization_results(self, model, results, predictions, features, strategy):
        """保存优化结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存模型
        model_path = os.path.join(self.optimization_dir, f'optimized_model_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'results': results,
                'features': features,
                'strategy': strategy,
                'timestamp': timestamp
            }, f)

        # 保存预测结果
        test_ids = self.test_features[['user_id', 'merchant_id']]
        submission = test_ids.copy()
        submission['prob'] = predictions

        pred_path = os.path.join(self.optimization_dir, f'optimized_predictions_{timestamp}.csv')
        submission.to_csv(pred_path, index=False)

        # 保存优化报告
        report = {
            'timestamp': timestamp,
            'optimization_results': results,
            'best_strategy': strategy,
            'selected_features': features,
            'feature_count': len(features),
            'performance_improvement': results['ensemble_auc'] - 0.6667,
            'base_directory': self.base_dir
        }

        report_path = os.path.join(self.optimization_dir, f'optimization_report_{timestamp}.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"优化结果已保存:")
        print(f"  模型: {model_path}")
        print(f"  预测: {pred_path}")
        print(f"  报告: {report_path}")


if __name__ == "__main__":
    print("开始模型性能优化 (D盘版本)...")

    # 使用D盘路径创建优化器
    optimizer = ModelOptimizerDDrive(
        feature_timestamp='20250910_051339',
        base_dir='D:/repeat_buyer_prediction'
    )

    # 运行优化流程
    optimized_model, results, predictions = optimizer.run_optimization_pipeline()

    print("模型优化完成！")