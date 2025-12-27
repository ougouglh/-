import pandas as pd
import numpy as np
import json
import os
import pickle
import warnings
from datetime import datetime
from collections import defaultdict

# 机器学习库
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, average_precision_score, precision_score,
    recall_score, f1_score, log_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

# 高级模型
import lightgbm as lgb
import xgboost as xgb

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedTimeAwareModelTrainer:
    """增强版时间感知模型训练器 - 支持集成学习和高级验证"""

    def __init__(self, feature_timestamp=None, output_dir='../outputs'):
        """
        初始化训练器

        Args:
            feature_timestamp: 增强特征的时间戳，如果为None则自动寻找最新的
            output_dir: 输出目录
        """
        self.feature_timestamp = feature_timestamp
        self.output_dir = output_dir

        # 设置路径
        self.feature_dir = os.path.join(output_dir, 'features_time_aware_enhanced')
        self.model_dir = os.path.join(output_dir, 'models_enhanced')
        self.results_dir = os.path.join(output_dir, 'results_enhanced')
        self.plots_dir = os.path.join(self.results_dir, 'plots')

        # 创建目录
        for dir_path in [self.model_dir, self.results_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 如果没有指定时间戳，自动寻找最新的
        if self.feature_timestamp is None:
            self.feature_timestamp = self._find_latest_feature_timestamp()

        print(f"增强版模型训练器初始化完成")
        print(f"使用特征时间戳: {self.feature_timestamp}")
        print(f"输出目录: {self.results_dir}")

    def _find_latest_feature_timestamp(self):
        """自动寻找最新的特征时间戳"""
        if not os.path.exists(self.feature_dir):
            raise FileNotFoundError(f"特征目录不存在: {self.feature_dir}")

        feature_files = [f for f in os.listdir(self.feature_dir) if f.startswith('train_features_enhanced_')]
        if not feature_files:
            raise FileNotFoundError(f"未找到增强特征文件在: {self.feature_dir}")

        # 提取时间戳
        timestamps = []
        for f in feature_files:
            try:
                timestamp = f.split('train_features_enhanced_')[1].split('.csv')[0]
                timestamps.append(timestamp)
            except:
                continue

        if not timestamps:
            raise ValueError("无法解析特征文件时间戳")

        return sorted(timestamps)[-1]

    def load_enhanced_features(self):
        """加载增强特征"""
        print("加载增强特征...")

        # 构建文件路径
        train_path = os.path.join(self.feature_dir, f'train_features_enhanced_{self.feature_timestamp}.csv')
        test_path = os.path.join(self.feature_dir, f'test_features_enhanced_{self.feature_timestamp}.csv')
        info_path = os.path.join(self.feature_dir, f'feature_info_enhanced_{self.feature_timestamp}.json')

        # 检查文件是否存在
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练特征文件不存在: {train_path}")

        # 加载数据
        self.train_features = pd.read_csv(train_path)
        self.test_features = pd.read_csv(test_path)

        print(f"训练特征: {self.train_features.shape}")
        print(f"测试特征: {self.test_features.shape}")

        # 加载特征信息
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                self.feature_info = json.load(f)
                print(f"特征数量: {self.feature_info['feature_count']}")
                print(f"增强功能: {', '.join(self.feature_info.get('enhancements', []))}")

        # 准备训练数据
        self._prepare_enhanced_training_data()

    def _prepare_enhanced_training_data(self):
        """准备增强训练数据"""
        print("准备增强训练数据...")

        # 分离特征和标签
        feature_cols = [col for col in self.train_features.columns
                        if col not in ['user_id', 'merchant_id', 'label']]

        self.X_train = self.train_features[feature_cols]
        self.y_train = self.train_features['label']
        self.train_ids = self.train_features[['user_id', 'merchant_id']]

        self.X_test = self.test_features[feature_cols]
        self.test_ids = self.test_features[['user_id', 'merchant_id']]

        self.feature_names = feature_cols

        # 数据清洗
        self.X_train = self.X_train.fillna(0).replace([np.inf, -np.inf], 0)
        self.X_test = self.X_test.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"最终训练数据: X{self.X_train.shape}, y{self.y_train.shape}")
        print(f"正样本比例: {self.y_train.mean() * 100:.2f}%")
        print(f"特征数量: {len(self.feature_names)}")

        # 特征预处理
        self._preprocess_features()

    # 在 modeltrainenh.py 中找到 _preprocess_features 方法（大约第150-170行）
    # 完全替换为以下代码：

    def _preprocess_features(self):
        """特征预处理"""
        print("特征预处理...")

        # 1. 移除低方差特征
        from sklearn.feature_selection import VarianceThreshold

        self.variance_selector = VarianceThreshold(threshold=0.0)  # 添加self.
        self.X_train_processed = self.variance_selector.fit_transform(self.X_train)
        self.X_test_processed = self.variance_selector.transform(self.X_test)

        # 更新特征名
        selected_features = self.variance_selector.get_support()
        self.processed_feature_names = [name for name, selected in zip(self.feature_names, selected_features) if
                                        selected]

        print(f"移除 {len(self.feature_names) - len(self.processed_feature_names)} 个低方差特征")
        print(f"剩余特征数量: {len(self.processed_feature_names)}")

        # 2. 特征缩放（为某些模型准备）
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_processed)
        self.X_test_scaled = self.scaler.transform(self.X_test_processed)

        # 转换回DataFrame便于使用
        self.X_train_processed = pd.DataFrame(self.X_train_processed, columns=self.processed_feature_names)
        self.X_test_processed = pd.DataFrame(self.X_test_processed, columns=self.processed_feature_names)

    def create_advanced_models(self, use_early_stopping=True):
        """创建高级模型配置"""
        print("创建高级模型配置...")

        # 计算类别权重
        pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])

        models = {
            'lightgbm': lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                metric='auc',
                is_unbalance=True,
                num_leaves=31,
                learning_rate=0.03,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                bagging_freq=5,
                min_child_samples=30,
                min_child_weight=1e-3,
                reg_alpha=0.2,
                reg_lambda=0.2,
                random_state=42,
                verbose=-1,
                n_estimators=250,
                early_stopping_rounds=50 if use_early_stopping else None
            ),

            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                scale_pos_weight=pos_weight,
                max_depth=6,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
            ),

            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),

            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                penalty='elasticnet',
                l1_ratio=0.5,
                solver='saga'
            )
        }

        return models

    def feature_selection(self, method='importance', k=None):
        """特征选择"""
        print(f"\n=== 特征选择: {method} ===")

        if k is None:
            k = min(150, len(self.processed_feature_names))  # 默认选择100个特征或全部特征

        if method == 'importance':
            # 基于重要性的特征选择
            temp_model = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=100,
                random_state=42,
                verbose=-1
            )
            temp_model.fit(self.X_train_processed, self.y_train)

            # 获取特征重要性
            importances = temp_model.feature_importances_
            indices = np.argsort(importances)[::-1][:k]

        elif method == 'univariate':
            # 单变量特征选择
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(self.X_train_processed, self.y_train)
            indices = selector.get_support(indices=True)

        elif method == 'rfe':
            # 递归特征消除
            temp_model = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=50,
                random_state=42,
                verbose=-1
            )
            rfe = RFE(estimator=temp_model, n_features_to_select=k, step=0.1)
            rfe.fit(self.X_train_processed, self.y_train)
            indices = np.where(rfe.support_)[0]

        else:
            # 默认使用所有特征
            indices = np.arange(len(self.processed_feature_names))

        # 更新特征
        self.selected_features = [self.processed_feature_names[i] for i in indices]
        self.X_train_selected = self.X_train_processed.iloc[:, indices]
        self.X_test_selected = self.X_test_processed.iloc[:, indices]

        print(f"选择了 {len(self.selected_features)} 个特征")

        return self.selected_features

    def advanced_cross_validation(self, models):
        """高级交叉验证"""
        print("\n=== 高级交叉验证 ===")

        # 创建多种验证策略
        cv_strategies = {
            'stratified_5fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            'stratified_10fold': StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        }

        results = {}

        for model_name, model in models.items():
            print(f"\n评估模型: {model_name}")

            model_results = {}

            for cv_name, cv in cv_strategies.items():
                print(f"  验证策略: {cv_name}")

                cv_scores = {
                    'auc': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'log_loss': []
                }

                # 存储每折的预测结果
                oof_predictions = np.zeros(len(self.y_train))

                for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train_selected, self.y_train)):
                    X_train_fold = self.X_train_selected.iloc[train_idx]
                    X_val_fold = self.X_train_selected.iloc[val_idx]
                    y_train_fold = self.y_train.iloc[train_idx]
                    y_val_fold = self.y_train.iloc[val_idx]

                    # 训练模型
                    if model_name == 'lightgbm':
                        model.fit(
                            X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)]
                        )
                    elif model_name == 'xgboost':
                        model.fit(
                            X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)],
                            verbose=False
                        )
                    else:
                        model.fit(X_train_fold, y_train_fold)

                    # 预测
                    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                    y_pred = model.predict(X_val_fold)

                    # 存储out-of-fold预测
                    oof_predictions[val_idx] = y_pred_proba

                    # 计算指标
                    cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
                    cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
                    cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
                    cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
                    cv_scores['log_loss'].append(log_loss(y_val_fold, y_pred_proba))

                # 计算平均性能
                mean_auc = np.mean(cv_scores['auc'])
                std_auc = np.std(cv_scores['auc'])

                model_results[cv_name] = {
                    'auc_mean': mean_auc,
                    'auc_std': std_auc,
                    'precision_mean': np.mean(cv_scores['precision']),
                    'recall_mean': np.mean(cv_scores['recall']),
                    'f1_mean': np.mean(cv_scores['f1']),
                    'log_loss_mean': np.mean(cv_scores['log_loss']),
                    'oof_predictions': oof_predictions,
                    'cv_scores': cv_scores
                }

                print(f"    AUC: {mean_auc:.4f} ± {std_auc:.4f}")

                # 性能合理性检查
                if mean_auc > 0.85:
                    print(f"    ⚠️ 警告: AUC过高 ({mean_auc:.4f})，可能仍有数据泄露")
                elif mean_auc < 0.55:
                    print(f"    ⚠️ 警告: AUC过低 ({mean_auc:.4f})，模型可能无效")
                else:
                    print(f"    ✅ AUC在合理范围内")

            results[model_name] = model_results

        self.cv_results = results
        return results

    def ensemble_training(self):
        """集成学习训练"""
        print("\n=== 集成学习训练 ===")

        # 获取最佳单模型的out-of-fold预测
        best_models = {}
        oof_predictions = {}

        for model_name in self.cv_results:
            # 选择最佳验证策略的结果
            best_cv = max(self.cv_results[model_name].keys(),
                          key=lambda k: self.cv_results[model_name][k]['auc_mean'])

            best_models[model_name] = self.cv_results[model_name][best_cv]
            oof_predictions[model_name] = best_models[model_name]['oof_predictions']

        # 1. 简单平均集成
        simple_avg = np.mean(list(oof_predictions.values()), axis=0)
        simple_avg_auc = roc_auc_score(self.y_train, simple_avg)
        print(f"简单平均AUC: {simple_avg_auc:.4f}")

        # 2. 加权平均集成（基于验证AUC）
        weights = np.array([best_models[name]['auc_mean'] for name in best_models])
        weights = weights / weights.sum()  # 标准化权重

        weighted_avg = np.zeros(len(self.y_train))
        for i, (model_name, predictions) in enumerate(oof_predictions.items()):
            weighted_avg += weights[i] * predictions
            print(f"  {model_name}: 权重 {weights[i]:.3f}")

        weighted_avg_auc = roc_auc_score(self.y_train, weighted_avg)
        print(f"加权平均AUC: {weighted_avg_auc:.4f}")

        # 3. Stacking集成
        stacking_auc = self._stacking_ensemble(oof_predictions)
        print(f"Stacking AUC: {stacking_auc:.4f}")

        # 4. Blending集成（留出验证集）
        blending_auc = self._blending_ensemble()
        print(f"Blending AUC: {blending_auc:.4f}")

        # 选择最佳集成方法
        ensemble_methods = {
            'simple_avg': simple_avg_auc,
            'weighted_avg': weighted_avg_auc,
            'stacking': stacking_auc,
            'blending': blending_auc
        }

        self.best_ensemble_method = max(ensemble_methods.keys(), key=lambda k: ensemble_methods[k])
        self.best_ensemble_auc = ensemble_methods[self.best_ensemble_method]

        print(f"\n最佳集成方法: {self.best_ensemble_method} (AUC: {self.best_ensemble_auc:.4f})")

        # 保存集成信息
        self.ensemble_info = {
            'method': self.best_ensemble_method,
            'auc': self.best_ensemble_auc,
            'weights': weights,
            'simple_avg_auc': simple_avg_auc,
            'weighted_avg_auc': weighted_avg_auc,
            'stacking_auc': stacking_auc,
            'blending_auc': blending_auc
        }

        return ensemble_methods

    def _stacking_ensemble(self, oof_predictions):
        """Stacking集成"""
        # 使用逻辑回归作为元学习器
        meta_features = np.column_stack(list(oof_predictions.values()))
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)

        # 交叉验证训练元学习器
        stacking_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        stacking_predictions = np.zeros(len(self.y_train))

        for train_idx, val_idx in stacking_cv.split(meta_features, self.y_train):
            meta_learner.fit(meta_features[train_idx], self.y_train.iloc[train_idx])
            stacking_predictions[val_idx] = meta_learner.predict_proba(meta_features[val_idx])[:, 1]

        self.meta_learner = meta_learner  # 保存元学习器
        return roc_auc_score(self.y_train, stacking_predictions)

    def _blending_ensemble(self):
        """Blending集成"""
        # 简化的blending实现
        X_blend, X_holdout, y_blend, y_holdout = train_test_split(
            self.X_train_selected, self.y_train, test_size=0.2,
            stratify=self.y_train, random_state=42
        )

        # 在blend集上训练模型（不使用早停）
        models = self.create_advanced_models(use_early_stopping=False)
        blend_predictions = []

        for model_name, model in models.items():
            try:
                if model_name == 'lightgbm':
                    model.fit(X_blend, y_blend)
                elif model_name == 'xgboost':
                    model.fit(X_blend, y_blend, verbose=False)
                else:
                    model.fit(X_blend, y_blend)

                pred = model.predict_proba(X_holdout)[:, 1]
                blend_predictions.append(pred)
            except Exception as e:
                print(f"Blending训练 {model_name} 失败: {e}")
                # 使用默认预测
                pred = np.full(len(X_holdout), y_blend.mean())
                blend_predictions.append(pred)

        if len(blend_predictions) > 0:
            # 简单平均
            ensemble_pred = np.mean(blend_predictions, axis=0)
            return roc_auc_score(y_holdout, ensemble_pred)
        else:
            return 0.5  # 默认AUC

    def train_final_models(self):
        """训练最终模型"""
        print(f"\n=== 训练最终模型 ===")

        # 训练所有基础模型
        models = self.create_advanced_models()
        self.final_models = {}

        for model_name, model in models.items():
            print(f"训练最终 {model_name} 模型...")

            if model_name == 'lightgbm':
                # 使用早停训练
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    self.X_train_selected, self.y_train, test_size=0.1,
                    stratify=self.y_train, random_state=42
                )

                model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val_split, y_val_split)]
                )
            elif model_name == 'xgboost':
                # 使用早停训练
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    self.X_train_selected, self.y_train, test_size=0.1,
                    stratify=self.y_train, random_state=42
                )

                model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val_split, y_val_split)],
                    verbose=False
                )
            else:
                model.fit(self.X_train_selected, self.y_train)

            self.final_models[model_name] = model

        # 如果最佳方法是stacking，训练最终的元学习器
        if self.best_ensemble_method == 'stacking':
            # 获取所有模型的预测
            all_predictions = []
            for model_name, model in self.final_models.items():
                pred = model.predict_proba(self.X_train_selected)[:, 1]
                all_predictions.append(pred)

            meta_features = np.column_stack(all_predictions)
            self.meta_learner.fit(meta_features, self.y_train)

        print("最终模型训练完成")

    def evaluate_final_models(self):
        """评估最终模型"""
        print("\n=== 最终模型评估 ===")

        # 评估每个基础模型
        for model_name, model in self.final_models.items():
            y_pred_proba = model.predict_proba(self.X_train_selected)[:, 1]
            auc = roc_auc_score(self.y_train, y_pred_proba)
            print(f"{model_name} AUC: {auc:.4f}")

        # 评估集成模型
        ensemble_pred = self._get_ensemble_prediction(self.X_train_selected)
        ensemble_auc = roc_auc_score(self.y_train, ensemble_pred)
        print(f"集成模型 AUC: {ensemble_auc:.4f}")

        # 预测分析
        print(f"\n集成预测分析:")
        print(f"  范围: {ensemble_pred.min():.6f} - {ensemble_pred.max():.6f}")
        print(f"  均值: {ensemble_pred.mean():.6f}")
        print(f"  中位数: {np.median(ensemble_pred):.6f}")

        # 按标签分析
        pos_probs = ensemble_pred[self.y_train == 1]
        neg_probs = ensemble_pred[self.y_train == 0]

        print(f"  正样本预测概率: 均值={pos_probs.mean():.4f}, 中位数={np.median(pos_probs):.4f}")
        print(f"  负样本预测概率: 均值={neg_probs.mean():.4f}, 中位数={np.median(neg_probs):.4f}")

        # 分离度检查
        separation = abs(pos_probs.mean() - neg_probs.mean())
        print(f"  正负样本分离度: {separation:.4f}")

        if separation < 0.1:
            print("  ⚠️ 注意: 分离度较低，模型区分能力有限")
        elif separation > 0.6:
            print("  ⚠️ 注意: 分离度很高，请检查是否有数据泄露")
        else:
            print("  ✅ 分离度正常")

        # 生成评估图表
        self._plot_comprehensive_evaluation(ensemble_pred)

        # 特征重要性分析
        self._analyze_comprehensive_feature_importance()

    def _get_ensemble_prediction(self, X):
        """获取集成预测"""
        if self.best_ensemble_method == 'simple_avg':
            predictions = []
            for model in self.final_models.values():
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)
            return np.mean(predictions, axis=0)

        elif self.best_ensemble_method == 'weighted_avg':
            predictions = []
            for model in self.final_models.values():
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)

            weighted_pred = np.zeros(len(X))
            for i, pred in enumerate(predictions):
                weighted_pred += self.ensemble_info['weights'][i] * pred
            return weighted_pred

        elif self.best_ensemble_method == 'stacking':
            # 获取所有基础模型预测
            base_predictions = []
            for model in self.final_models.values():
                pred = model.predict_proba(X)[:, 1]
                base_predictions.append(pred)

            meta_features = np.column_stack(base_predictions)
            return self.meta_learner.predict_proba(meta_features)[:, 1]

        else:  # blending 或其他方法，默认简单平均
            predictions = []
            for model in self.final_models.values():
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)
            return np.mean(predictions, axis=0)

    def _plot_comprehensive_evaluation(self, y_pred_proba):
        """绘制综合评估图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # ROC曲线
        fpr, tpr, _ = roc_curve(self.y_train, y_pred_proba)
        auc_score = roc_auc_score(self.y_train, y_pred_proba)

        axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', linewidth=2)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('假正率')
        axes[0, 0].set_ylabel('真正率')
        axes[0, 0].set_title('ROC曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # PR曲线
        precision, recall, _ = precision_recall_curve(self.y_train, y_pred_proba)
        ap_score = average_precision_score(self.y_train, y_pred_proba)

        axes[0, 1].plot(recall, precision, label=f'AP = {ap_score:.4f}', linewidth=2)
        axes[0, 1].set_xlabel('召回率')
        axes[0, 1].set_ylabel('精确率')
        axes[0, 1].set_title('Precision-Recall曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 预测概率分布
        axes[0, 2].hist(y_pred_proba[self.y_train == 0], bins=50, alpha=0.7,
                        label='负样本', density=True, color='blue')
        axes[0, 2].hist(y_pred_proba[self.y_train == 1], bins=50, alpha=0.7,
                        label='正样本', density=True, color='red')
        axes[0, 2].set_xlabel('预测概率')
        axes[0, 2].set_ylabel('密度')
        axes[0, 2].set_title('预测概率分布')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 校准曲线
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_train, y_pred_proba, n_bins=10
        )

        axes[1, 0].plot(mean_predicted_value, fraction_of_positives, "s-", label="模型")
        axes[1, 0].plot([0, 1], [0, 1], "k:", label="完美校准")
        axes[1, 0].set_xlabel('平均预测概率')
        axes[1, 0].set_ylabel('实际正样本比例')
        axes[1, 0].set_title('校准曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 阈值-性能曲线
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1_scores.append(f1_score(self.y_train, y_pred, zero_division=0))
            precisions.append(precision_score(self.y_train, y_pred, zero_division=0))
            recalls.append(recall_score(self.y_train, y_pred, zero_division=0))

        axes[1, 1].plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        axes[1, 1].plot(thresholds, precisions, label='Precision', linewidth=2)
        axes[1, 1].plot(thresholds, recalls, label='Recall', linewidth=2)
        axes[1, 1].set_xlabel('阈值')
        axes[1, 1].set_ylabel('性能指标')
        axes[1, 1].set_title('阈值-性能曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 混淆矩阵（使用最佳F1阈值）
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        y_pred_best = (y_pred_proba >= best_threshold).astype(int)

        cm = confusion_matrix(self.y_train, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2])
        axes[1, 2].set_xlabel('预测标签')
        axes[1, 2].set_ylabel('真实标签')
        axes[1, 2].set_title(f'混淆矩阵 (阈值={best_threshold:.2f})')

        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(self.plots_dir, 'comprehensive_model_evaluation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"综合评估图表已保存: {plot_path}")

    def _analyze_comprehensive_feature_importance(self):
        """综合特征重要性分析"""
        print("\n=== 综合特征重要性分析 ===")

        # 收集所有模型的特征重要性
        all_importances = {}

        for model_name, model in self.final_models.items():
            if hasattr(model, 'feature_importances_'):
                # 树模型的特征重要性
                importances = model.feature_importances_
                all_importances[model_name] = importances
            elif hasattr(model, 'coef_'):
                # 线性模型的系数
                importances = np.abs(model.coef_[0])
                all_importances[model_name] = importances

        if all_importances:
            # 计算平均重要性
            avg_importance = np.mean(list(all_importances.values()), axis=0)

            # 创建特征重要性DataFrame
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)

            # 添加各模型的重要性
            for model_name, importances in all_importances.items():
                feature_importance[f'{model_name}_importance'] = importances

            print("TOP 15 重要特征:")
            for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
                print(f"  {i + 1:2d}. {row['feature']:40s} {row['importance']:.4f}")

            # 特征重要性可视化
            self._plot_feature_importance(feature_importance)

            # 保存特征重要性
            importance_path = os.path.join(self.results_dir, 'comprehensive_feature_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            print(f"特征重要性已保存: {importance_path}")

            # 特征类型分析
            self._analyze_feature_types(feature_importance)

    def _plot_feature_importance(self, feature_importance):
        """绘制特征重要性图"""
        plt.figure(figsize=(12, 8))

        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性')
        plt.title('TOP 20 特征重要性')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(self.plots_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_feature_types(self, feature_importance):
        """分析特征类型的重要性分布"""
        print("\n特征类型重要性分析:")

        # 定义特征类型
        feature_types = {
            'user_basic': [f for f in self.selected_features if f.startswith('user_hist_')],
            'user_enhanced': [f for f in self.selected_features if
                              f.startswith('user_') and not f.startswith('user_hist_')],
            'merchant_basic': [f for f in self.selected_features if f.startswith('merchant_hist_')],
            'merchant_enhanced': [f for f in self.selected_features if
                                  f.startswith('merchant_') and not f.startswith('merchant_hist_')],
            'interaction': [f for f in self.selected_features if f.startswith('interaction_')],
            'profile': [f for f in self.selected_features if any(x in f for x in ['age', 'gender'])],
            'statistical': [f for f in self.selected_features if '_stat_' in f],
            'log_features': [f for f in self.selected_features if f.endswith('_log')]
        }

        for feature_type, features in feature_types.items():
            if features:
                type_importances = feature_importance[feature_importance['feature'].isin(features)]['importance']
                avg_importance = type_importances.mean()
                total_importance = type_importances.sum()
                feature_count = len(features)

                print(
                    f"  {feature_type:20s}: {feature_count:3d}个特征, 平均重要性={avg_importance:.4f}, 总重要性={total_importance:.4f}")

    def generate_test_predictions(self):
        """生成测试集预测"""
        print("\n=== 生成测试集预测 ===")

        # 使用集成模型预测测试集
        test_pred_proba = self._get_ensemble_prediction(self.X_test_selected)

        # 分析预测结果
        print("测试集预测分析:")
        print(f"  预测概率范围: {test_pred_proba.min():.6f} - {test_pred_proba.max():.6f}")
        print(f"  预测概率均值: {test_pred_proba.mean():.6f}")
        print(f"  预测概率中位数: {np.median(test_pred_proba):.6f}")

        # 预期检查
        expected_positive_rate = self.y_train.mean()
        high_prob_rate = np.mean(test_pred_proba > 0.5)

        print(f"  训练集正样本率: {expected_positive_rate * 100:.2f}%")
        print(f"  测试集高概率样本率 (>0.5): {high_prob_rate * 100:.2f}%")

        # 概率分布分析
        prob_percentiles = np.percentile(test_pred_proba, [10, 25, 50, 75, 90, 95, 99])
        print(f"  概率分布百分位: P10={prob_percentiles[0]:.4f}, P25={prob_percentiles[1]:.4f}, "
              f"P50={prob_percentiles[2]:.4f}, P75={prob_percentiles[3]:.4f}, P90={prob_percentiles[4]:.4f}, "
              f"P95={prob_percentiles[5]:.4f}, P99={prob_percentiles[6]:.4f}")

        # 合理性检查
        if np.all(test_pred_proba < 0.001):
            print("  ⚠️ 警告: 所有预测概率都很低，可能有问题")
        elif np.all(test_pred_proba > 0.999):
            print("  ⚠️ 警告: 所有预测概率都很高，可能有问题")
        elif np.std(test_pred_proba) < 0.01:
            print("  ⚠️ 警告: 预测概率方差很小，模型可能没有学到有效模式")
        else:
            print("  ✅ 预测概率分布正常")

        # 创建提交文件
        submission = self.test_ids.copy()
        submission['prob'] = test_pred_proba

        return submission

    def save_comprehensive_results(self, submission):
        """保存综合结果"""
        print("\n=== 保存综合结果 ===")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存预测结果
        submission_path = os.path.join(self.results_dir, f'prediction_enhanced_{timestamp}.csv')
        submission.to_csv(submission_path, index=False)

        # 保存模型
        model_info = {
            'timestamp': timestamp,
            'feature_timestamp': self.feature_timestamp,
            'final_models': self.final_models,
            'processed_feature_names': self.processed_feature_names,
            'variance_selector': getattr(self, 'variance_selector', None),
            'meta_learner': getattr(self, 'meta_learner', None),
            'ensemble_info': self.ensemble_info,
            'cv_results': self.cv_results,
            'selected_features': self.selected_features,
            'scaler': self.scaler,
            'best_ensemble_method': self.best_ensemble_method,
            'best_ensemble_auc': self.best_ensemble_auc
        }

        model_path = os.path.join(self.model_dir, f'enhanced_model_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)

        # 保存训练报告（处理numpy数组序列化问题）
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj

        report = {
            'timestamp': timestamp,
            'feature_timestamp': self.feature_timestamp,
            'final_feature_count': len(self.selected_features),
            'best_ensemble_method': self.best_ensemble_method,
            'best_ensemble_auc': float(self.best_ensemble_auc),
            'ensemble_performance': convert_numpy_types(self.ensemble_info),
            'cross_validation_results': {
                model_name: {cv_name: float(result['auc_mean']) for cv_name, result in model_cv.items()}
                for model_name, model_cv in self.cv_results.items()
            },
            'prediction_stats': {
                'min': float(submission['prob'].min()),
                'max': float(submission['prob'].max()),
                'mean': float(submission['prob'].mean()),
                'std': float(submission['prob'].std()),
                'high_prob_rate': float(np.mean(submission['prob'] > 0.5))
            }
        }

        report_path = os.path.join(self.results_dir, f'training_report_{timestamp}.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"结果已保存:")
        print(f"  预测文件: {submission_path}")
        print(f"  模型文件: {model_path}")
        print(f"  训练报告: {report_path}")

        return timestamp

    def run_complete_enhanced_pipeline(self):
        """运行完整的增强模型训练流程"""
        print("=== 开始增强版模型训练流程 ===")

        try:
            # 1. 加载增强特征
            self.load_enhanced_features()

            # 2. 特征选择
            self.feature_selection(method='importance', k=100)

            # 3. 创建高级模型
            models = self.create_advanced_models()

            # 4. 高级交叉验证
            cv_results = self.advanced_cross_validation(models)

            # 5. 集成学习
            ensemble_results = self.ensemble_training()

            # 6. 训练最终模型
            self.train_final_models()

            # 7. 评估最终模型
            self.evaluate_final_models()

            # 8. 生成测试集预测
            submission = self.generate_test_predictions()

            # 9. 保存综合结果
            result_timestamp = self.save_comprehensive_results(submission)

            print(f"\n=== 增强版模型训练完成 ===")
            print(f"最佳集成方法: {self.best_ensemble_method}")
            print(f"最佳AUC: {self.best_ensemble_auc:.4f}")
            print(f"使用特征数: {len(self.selected_features)}")
            print(f"结果时间戳: {result_timestamp}")
            print(f"预测文件: prediction_enhanced_{result_timestamp}.csv")

            return submission, result_timestamp

        except Exception as e:
            print(f"训练过程出错: {e}")
            import traceback
            traceback.print_exc()
            raise


# 兼容性包装器
class TimeAwareModelTrainer(EnhancedTimeAwareModelTrainer):
    """原始接口的兼容性包装器"""

    def __init__(self, feature_timestamp=None, output_dir='../outputs'):
        super().__init__(feature_timestamp, output_dir)
        print("注意: 正在使用增强版模型训练器")

    def run_complete_pipeline(self):
        """运行完整流程（原始接口）"""
        return self.run_complete_enhanced_pipeline()


if __name__ == "__main__":
    print("开始增强版模型训练...")

    try:
        # 创建增强版训练器
        trainer = EnhancedTimeAwareModelTrainer()

        # 运行完整流程
        submission, timestamp = trainer.run_complete_enhanced_pipeline()

        print("增强版模型训练完成！")
        print(f"最佳模型AUC: {trainer.best_ensemble_auc:.4f}")
        print(f"请使用 prediction_enhanced_{timestamp}.csv 提交结果")

    except Exception as e:
        print(f"训练失败: {e}")
        print("请检查:")
        print("1. 增强特征文件是否存在")
        print("2. 数据路径是否正确")
        print("3. 依赖包是否完整安装")