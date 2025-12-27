import pandas as pd
import numpy as np
import json
import os
import pickle
import warnings
from datetime import datetime

# 机器学习库
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, average_precision_score, precision_score,
    recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# 高级模型
import lightgbm as lgb
import xgboost as xgb

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TimeAwareModelTrainer:
    """基于时间感知特征的模型训练器"""

    def __init__(self, feature_timestamp='20250910_051339', output_dir='../outputs'):
        """
        初始化训练器

        Args:
            feature_timestamp: 时间感知特征的时间戳
            output_dir: 输出目录
        """
        self.feature_timestamp = feature_timestamp
        self.output_dir = output_dir

        # 设置路径
        self.feature_dir = os.path.join(output_dir, 'features_time_aware')
        self.model_dir = os.path.join(output_dir, 'models_time_aware')
        self.results_dir = os.path.join(output_dir, 'results_time_aware')
        self.plots_dir = os.path.join(self.results_dir, 'plots')

        # 创建目录
        for dir_path in [self.model_dir, self.results_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)

        print(f"时间感知模型训练器初始化完成")
        print(f"使用特征时间戳: {feature_timestamp}")
        print(f"输出目录: {self.results_dir}")

    def load_time_aware_features(self):
        """加载时间感知特征"""
        print("加载时间感知特征...")

        # 构建文件路径
        train_path = os.path.join(self.feature_dir, f'train_features_time_aware_{self.feature_timestamp}.csv')
        test_path = os.path.join(self.feature_dir, f'test_features_time_aware_{self.feature_timestamp}.csv')
        info_path = os.path.join(self.feature_dir, f'feature_info_time_aware_{self.feature_timestamp}.json')

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

        # 准备训练数据
        self._prepare_training_data()

    def _prepare_training_data(self):
        """准备训练数据"""
        print("准备训练数据...")

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

    def create_time_aware_validation(self):
        """创建时间感知的验证策略"""
        print("创建时间感知验证策略...")

        # 对于时间序列数据，我们使用分层抽样但更谨慎
        # 由于我们已经在特征工程阶段处理了时间问题，这里使用分层抽样是安全的
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        return cv

    def quick_model_evaluation(self):
        """快速模型评估"""
        print("\n=== 快速模型评估 ===")

        # 简单模型集合
        models = {
            'logistic': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                objective='binary',
                class_weight='balanced',
                random_state=42,
                verbose=-1,
                n_estimators=100
            ),
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            )
        }

        # 交叉验证
        cv = self.create_time_aware_validation()
        results = {}

        for model_name, model in models.items():
            print(f"\n评估模型: {model_name}")

            cv_scores = {
                'auc': [],
                'precision': [],
                'recall': [],
                'f1': []
            }

            for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
                X_train_fold = self.X_train.iloc[train_idx]
                X_val_fold = self.X_train.iloc[val_idx]
                y_train_fold = self.y_train.iloc[train_idx]
                y_val_fold = self.y_train.iloc[val_idx]

                # 训练模型
                model.fit(X_train_fold, y_train_fold)

                # 预测
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                y_pred = model.predict(X_val_fold)

                # 计算指标
                cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
                cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
                cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
                cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))

            # 计算平均性能
            mean_auc = np.mean(cv_scores['auc'])
            std_auc = np.std(cv_scores['auc'])

            results[model_name] = {
                'auc_mean': mean_auc,
                'auc_std': std_auc,
                'precision_mean': np.mean(cv_scores['precision']),
                'recall_mean': np.mean(cv_scores['recall']),
                'f1_mean': np.mean(cv_scores['f1'])
            }

            print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}")

            # 性能合理性检查
            if mean_auc > 0.90:
                print(f"  ⚠️ 警告: AUC过高 ({mean_auc:.4f})，可能仍有数据泄露")
            elif mean_auc < 0.55:
                print(f"  ⚠️ 警告: AUC过低 ({mean_auc:.4f})，模型可能无效")
            else:
                print(f"  ✅ AUC在合理范围内")

        # 选择最佳模型
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_mean'])
        best_auc = results[best_model_name]['auc_mean']

        print(f"\n最佳模型: {best_model_name} (AUC: {best_auc:.4f})")

        self.best_model_name = best_model_name
        self.evaluation_results = results

        return results

    def train_final_model(self):
        """训练最终模型"""
        print(f"\n=== 训练最终模型: {self.best_model_name} ===")

        # 创建最终模型
        if self.best_model_name == 'lightgbm':
            self.final_model = lgb.LGBMClassifier(
                objective='binary',
                class_weight='balanced',
                random_state=42,
                verbose=-1,
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20
            )
        elif self.best_model_name == 'xgboost':
            # 计算类别权重
            pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
            self.final_model = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=pos_weight,
                random_state=42,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6
            )
        elif self.best_model_name == 'random_forest':
            self.final_model = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_estimators=200,
                max_depth=10,
                n_jobs=-1
            )
        else:
            self.final_model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )

        # 训练模型
        print("训练最终模型...")
        self.final_model.fit(self.X_train, self.y_train)

        # 评估模型
        self._evaluate_final_model()

        # 保存模型
        self._save_model()

    def _evaluate_final_model(self):
        """评估最终模型"""
        print("\n模型评估:")

        # 训练集预测
        y_train_pred_proba = self.final_model.predict_proba(self.X_train)[:, 1]
        y_train_pred = self.final_model.predict(self.X_train)

        # 计算指标
        train_auc = roc_auc_score(self.y_train, y_train_pred_proba)
        train_ap = average_precision_score(self.y_train, y_train_pred_proba)

        print(f"训练集AUC: {train_auc:.4f}")
        print(f"训练集AP: {train_ap:.4f}")

        # 预测概率分析
        print(f"\n预测概率分析:")
        print(f"  范围: {y_train_pred_proba.min():.6f} - {y_train_pred_proba.max():.6f}")
        print(f"  均值: {y_train_pred_proba.mean():.6f}")
        print(f"  中位数: {np.median(y_train_pred_proba):.6f}")

        # 按标签分析
        pos_probs = y_train_pred_proba[self.y_train == 1]
        neg_probs = y_train_pred_proba[self.y_train == 0]

        print(f"  正样本预测概率: 均值={pos_probs.mean():.4f}, 中位数={np.median(pos_probs):.4f}")
        print(f"  负样本预测概率: 均值={neg_probs.mean():.4f}, 中位数={np.median(neg_probs):.4f}")

        # 分离度检查
        separation = abs(pos_probs.mean() - neg_probs.mean())
        print(f"  正负样本分离度: {separation:.4f}")

        if separation < 0.1:
            print("  ⚠️ 注意: 分离度较低，模型区分能力有限")
        elif separation > 0.8:
            print("  ⚠️ 注意: 分离度很高，请检查是否有数据泄露")
        else:
            print("  ✅ 分离度正常")

        # 可视化
        self._plot_model_evaluation(self.y_train, y_train_pred_proba)

        # 特征重要性
        self._analyze_feature_importance()

    def _plot_model_evaluation(self, y_true, y_pred_proba):
        """绘制模型评估图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)

        axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('假正率')
        axes[0, 0].set_ylabel('真正率')
        axes[0, 0].set_title('ROC曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)

        axes[0, 1].plot(recall, precision, label=f'AP = {ap_score:.4f}')
        axes[0, 1].set_xlabel('召回率')
        axes[0, 1].set_ylabel('精确率')
        axes[0, 1].set_title('Precision-Recall曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 预测概率分布
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='负样本', density=True)
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='正样本', density=True)
        axes[1, 0].set_xlabel('预测概率')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].set_title('预测概率分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 概率vs标签散点图（采样）
        sample_size = min(10000, len(y_true))
        sample_idx = np.random.choice(len(y_true), sample_size, replace=False)

        axes[1, 1].scatter(y_pred_proba[sample_idx], y_true[sample_idx], alpha=0.3)
        axes[1, 1].set_xlabel('预测概率')
        axes[1, 1].set_ylabel('真实标签')
        axes[1, 1].set_title('预测概率 vs 真实标签')
        axes[1, 1].grid(True)

        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(self.plots_dir, 'time_aware_model_evaluation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"评估图表已保存: {plot_path}")

    def _analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n特征重要性分析:")

        if hasattr(self.final_model, 'feature_importances_'):
            # 树模型的特征重要性
            importances = self.final_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print("TOP 10 重要特征:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"  {i + 1:2d}. {row['feature']:40s} {row['importance']:.4f}")

            # 保存特征重要性
            importance_path = os.path.join(self.results_dir, 'feature_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            print(f"特征重要性已保存: {importance_path}")

        elif hasattr(self.final_model, 'coef_'):
            # 线性模型的系数
            coefs = abs(self.final_model.coef_[0])
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'abs_coefficient': coefs
            }).sort_values('abs_coefficient', ascending=False)

            print("TOP 10 重要特征 (按系数绝对值):")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"  {i + 1:2d}. {row['feature']:40s} {row['abs_coefficient']:.4f}")

    def _save_model(self):
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f'time_aware_model_{self.best_model_name}_{timestamp}.pkl')

        model_info = {
            'model': self.final_model,
            'feature_names': self.feature_names,
            'feature_timestamp': self.feature_timestamp,
            'model_name': self.best_model_name,
            'evaluation_results': self.evaluation_results
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)

        print(f"模型已保存: {model_path}")

    def generate_predictions(self):
        """生成测试集预测"""
        print("\n=== 生成测试集预测 ===")

        # 预测测试集
        test_pred_proba = self.final_model.predict_proba(self.X_test)[:, 1]

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

        # 合理性检查
        if np.all(test_pred_proba < 0.001):
            print("  ⚠️ 警告: 所有预测概率都很低，可能有问题")
        elif np.all(test_pred_proba > 0.999):
            print("  ⚠️ 警告: 所有预测概率都很高，可能有问题")
        else:
            print("  ✅ 预测概率分布正常")

        # 创建提交文件
        submission = self.test_ids.copy()
        submission['prob'] = test_pred_proba

        # 保存预测结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        submission_path = os.path.join(self.results_dir, f'prediction_time_aware_{timestamp}.csv')
        submission.to_csv(submission_path, index=False)

        print(f"预测结果已保存: {submission_path}")

        return submission

    def run_complete_pipeline(self):
        """运行完整的时间感知模型训练流程"""
        print("=== 开始时间感知模型训练流程 ===")

        # 1. 加载时间感知特征
        self.load_time_aware_features()

        # 2. 快速模型评估
        evaluation_results = self.quick_model_evaluation()

        # 3. 训练最终模型
        self.train_final_model()

        # 4. 生成预测
        submission = self.generate_predictions()

        print(f"\n=== 时间感知模型训练完成 ===")
        print(f"最佳模型: {self.best_model_name}")
        print(f"预期性能: AUC {self.evaluation_results[self.best_model_name]['auc_mean']:.4f}")
        print(f"结果保存在: {self.results_dir}")

        return submission


if __name__ == "__main__":
    print("开始基于时间感知特征的模型训练...")

    # 使用您的特征时间戳
    trainer = TimeAwareModelTrainer(feature_timestamp='20250910_051339')

    # 运行完整流程
    submission = trainer.run_complete_pipeline()

    print("时间感知模型训练完成！")