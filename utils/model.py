import pandas as pd
import numpy as np
import json
import os
import pickle
import warnings
from datetime import datetime

# 机器学习库
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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


class FixedRepeatBuyerTrainer:
    """修复版重复购买预测模型训练器"""

    def __init__(self, feature_dir='../outputs/features', output_dir='../outputs'):
        self.feature_dir = feature_dir
        self.output_dir = output_dir

        # 创建输出目录
        self.model_dir = os.path.join(output_dir, 'models_fixed')
        self.results_dir = os.path.join(output_dir, 'model_results_fixed')
        self.plots_dir = os.path.join(self.results_dir, 'plots')

        for dir_path in [self.model_dir, self.results_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 数据容器
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.train_ids = None
        self.test_ids = None
        self.feature_names = None

        print(f"修复版训练器初始化完成")
        print(f"输出目录: {self.results_dir}")

    def load_data_with_proper_features(self):
        """加载数据并正确选择特征"""
        print("加载数据并修复特征选择...")

        # 找到最新的特征文件
        feature_files = [f for f in os.listdir(self.feature_dir)
                         if f.startswith('train_features_') and f.endswith('.csv')]
        if not feature_files:
            raise FileNotFoundError("未找到特征文件")

        latest_file = sorted(feature_files)[-1]
        timestamp = latest_file.replace('train_features_', '').replace('.csv', '')
        print(f"使用特征文件: {timestamp}")

        # 加载特征文件
        train_features_path = os.path.join(self.feature_dir, f'train_features_{timestamp}.csv')
        test_features_path = os.path.join(self.feature_dir, f'test_features_{timestamp}.csv')

        train_features = pd.read_csv(train_features_path)
        test_features = pd.read_csv(test_features_path)

        print(f"原始特征: 训练{train_features.shape}, 测试{test_features.shape}")

        # 加载原始标签
        train_original = pd.read_csv('../data/data_format1/train_format1.csv')
        train_data = train_features.merge(
            train_original[['user_id', 'merchant_id', 'label']],
            on=['user_id', 'merchant_id'],
            how='left'
        )

        # 定义安全的特征集（根据您之前的特征重要性分析）
        # 排除明显的泄露特征和ID特征
        exclude_features = [
            'user_id', 'merchant_id', 'label', 'prob',  # ID和目标相关
            'Unnamed', 'index'  # 索引相关
        ]

        # 基于您之前分析的TOP特征（手动指定以避免泄露）
        recommended_features = [
            'user_repeat_count', 'user_repeat_rate', 'user_loyalty_score',
            'merchant_repeat_rate', 'merchant_repeat_std', 'merchant_repeat_stability',
            'user_total_merchants', 'user_unique_merchants',
            'merchant_total_transactions', 'merchant_unique_users',
            'loyalty_vs_merchant_rate', 'diversity_vs_size',
            'user_merchant_relative_activity', 'merchant_size_score',
            'total_actions', 'action_type_diversity', 'unique_items',
            'click_to_buy_rate', 'item_diversity', 'avg_actions_per_time',
            'active_time_points', 'loyalty_x_large_merchant',
            'diversity_x_repeat_rate', 'activity_x_merchant_size',
            'user_relative_activity'
        ]

        # 筛选实际存在的特征
        available_features = []
        for feature in recommended_features:
            if feature in train_features.columns:
                available_features.append(feature)

        # 如果推荐特征不够，从剩余安全特征中补充
        if len(available_features) < 20:
            all_features = [col for col in train_features.columns
                            if not any(exc in col for exc in exclude_features)]
            for feature in all_features:
                if feature not in available_features and len(available_features) < 25:
                    available_features.append(feature)

        print(f"使用{len(available_features)}个安全特征:")
        for i, feature in enumerate(available_features[:20], 1):  # 显示前20个
            print(f"  {i:2d}. {feature}")
        if len(available_features) > 20:
            print(f"  ... 还有{len(available_features) - 20}个特征")

        # 准备数据
        self.X_train = train_data[available_features]
        self.y_train = train_data['label']
        self.train_ids = train_data[['user_id', 'merchant_id']]

        self.X_test = test_features[available_features]
        self.test_ids = test_features[['user_id', 'merchant_id']]

        self.feature_names = available_features

        # 数据清洗
        self.X_train = self.X_train.fillna(0).replace([np.inf, -np.inf], 0)
        self.X_test = self.X_test.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"最终数据: 训练X{self.X_train.shape}, y{self.y_train.shape}")
        print(f"正样本比例: {self.y_train.mean() * 100:.2f}%")

        # 合理性检查
        self._sanity_check()

    def _sanity_check(self):
        """数据合理性检查"""
        print("\n数据合理性检查:")

        # 检查特征分布
        feature_stats = self.X_train.describe()
        print(f"特征统计摘要:")
        print(f"  特征数量: {len(self.feature_names)}")
        print(f"  特征值范围: {self.X_train.min().min():.6f} - {self.X_train.max().max():.6f}")

        # 检查是否有明显异常的特征
        suspicious_features = []
        for col in self.X_train.columns:
            unique_ratio = self.X_train[col].nunique() / len(self.X_train)
            if unique_ratio > 0.9:  # 唯一值比例过高
                suspicious_features.append(f"{col} (唯一值比例: {unique_ratio:.3f})")

        if suspicious_features:
            print("  潜在问题特征:")
            for feature in suspicious_features[:5]:
                print(f"    {feature}")

        print("合理性检查完成")

    def create_balanced_datasets(self):
        """创建平衡数据集"""
        print("\n创建平衡数据集...")

        self.datasets = {}

        # 原始数据
        self.datasets['original'] = (self.X_train, self.y_train)
        print(f"原始数据: {self.X_train.shape}, 正样本率: {self.y_train.mean() * 100:.2f}%")

        # 欠采样版本（推荐用于初始测试）
        pos_indices = self.y_train[self.y_train == 1].index
        neg_indices = self.y_train[self.y_train == 0].index

        # 适度欠采样：保留更多负样本
        neg_sample_size = min(len(pos_indices) * 5, len(neg_indices))  # 1:5比例
        neg_sampled = resample(neg_indices,
                               replace=False,
                               n_samples=neg_sample_size,
                               random_state=42)

        balanced_indices = np.concatenate([pos_indices, neg_sampled])
        X_balanced = self.X_train.loc[balanced_indices]
        y_balanced = self.y_train.loc[balanced_indices]

        self.datasets['balanced'] = (X_balanced, y_balanced)
        print(f"平衡数据: {X_balanced.shape}, 正样本率: {y_balanced.mean() * 100:.2f}%")

    def quick_model_comparison(self):
        """快速模型比较"""
        print("\n快速模型比较...")

        # 简单模型配置
        models = {
            'logistic': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
            'lightgbm': lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1, n_estimators=100)
        }

        results = {}

        for dataset_name, (X, y) in self.datasets.items():
            print(f"\n数据集: {dataset_name}")
            results[dataset_name] = {}

            # 3折交叉验证
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            for model_name, model in models.items():
                auc_scores = []

                for train_idx, val_idx in skf.split(X, y):
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    y_val_fold = y.iloc[val_idx]

                    # 训练
                    model.fit(X_train_fold, y_train_fold)

                    # 预测
                    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                    auc_score = roc_auc_score(y_val_fold, y_pred_proba)
                    auc_scores.append(auc_score)

                mean_auc = np.mean(auc_scores)
                std_auc = np.std(auc_scores)
                results[dataset_name][model_name] = mean_auc

                print(f"  {model_name}: AUC = {mean_auc:.4f} ± {std_auc:.4f}")

        # 选择最佳组合
        best_auc = 0
        best_combo = None

        for dataset_name, models_result in results.items():
            for model_name, auc in models_result.items():
                if auc > best_auc:
                    best_auc = auc
                    best_combo = (dataset_name, model_name)

        print(f"\n最佳组合: {best_combo[1]} + {best_combo[0]} (AUC: {best_auc:.4f})")

        # 检查AUC是否合理
        if best_auc > 0.95:
            print("⚠️  警告: AUC过高，可能仍存在数据泄露问题!")
            print("   建议进一步检查特征工程和数据质量")
        elif best_auc < 0.55:
            print("⚠️  警告: AUC过低，模型可能无效")
            print("   建议检查特征质量和模型配置")
        else:
            print("✅ AUC在合理范围内")

        self.best_dataset, self.best_model_name = best_combo
        self.best_auc = best_auc

        return results

    def train_final_model(self):
        """训练最终模型"""
        print(f"\n训练最终模型: {self.best_model_name} on {self.best_dataset}")

        # 获取最佳数据集
        X, y = self.datasets[self.best_dataset]

        # 创建模型
        if self.best_model_name == 'lightgbm':
            self.final_model = lgb.LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                verbose=-1,
                n_estimators=200,
                learning_rate=0.1
            )
        elif self.best_model_name == 'random_forest':
            self.final_model = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_estimators=200
            )
        else:
            self.final_model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )

        # 训练
        self.final_model.fit(X, y)

        # 评估
        self._evaluate_model(X, y)

    def _evaluate_model(self, X, y):
        """评估模型"""
        print("\n模型评估:")

        # 预测
        y_pred_proba = self.final_model.predict_proba(X)[:, 1]
        y_pred = self.final_model.predict(X)

        # 计算指标
        auc_score = roc_auc_score(y, y_pred_proba)
        ap_score = average_precision_score(y, y_pred_proba)

        print(f"AUC: {auc_score:.4f}")
        print(f"AP: {ap_score:.4f}")

        # 预测概率分析
        print(f"预测概率统计:")
        print(f"  最小值: {y_pred_proba.min():.6f}")
        print(f"  最大值: {y_pred_proba.max():.6f}")
        print(f"  平均值: {y_pred_proba.mean():.6f}")
        print(f"  中位数: {np.median(y_pred_proba):.6f}")

        # 按真实标签分组的概率分布
        pos_probs = y_pred_proba[y == 1]
        neg_probs = y_pred_proba[y == 0]

        print(f"正样本预测概率: 均值={pos_probs.mean():.4f}, 中位数={np.median(pos_probs):.4f}")
        print(f"负样本预测概率: 均值={neg_probs.mean():.4f}, 中位数={np.median(neg_probs):.4f}")

        # 可视化
        self._plot_evaluation(y, y_pred_proba)

    def _plot_evaluation(self, y_true, y_pred_proba):
        """绘制评估图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)

        axes[0, 0].plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('假正率 (FPR)')
        axes[0, 0].set_ylabel('真正率 (TPR)')
        axes[0, 0].set_title('ROC曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)

        axes[0, 1].plot(recall, precision, label=f'AP = {ap_score:.4f}')
        axes[0, 1].set_xlabel('召回率 (Recall)')
        axes[0, 1].set_ylabel('精确率 (Precision)')
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

        # 概率分位数图
        prob_percentiles = np.percentile(y_pred_proba, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        axes[1, 1].bar(range(len(prob_percentiles)), prob_percentiles)
        axes[1, 1].set_xticks(range(len(prob_percentiles)))
        axes[1, 1].set_xticklabels(['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%'])
        axes[1, 1].set_xlabel('分位数')
        axes[1, 1].set_ylabel('预测概率')
        axes[1, 1].set_title('预测概率分位数')
        axes[1, 1].grid(True)

        plt.tight_layout()

        # 保存
        plot_path = os.path.join(self.plots_dir, 'model_evaluation_fixed.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"评估图表已保存: {plot_path}")

    def generate_predictions(self):
        """生成预测结果"""
        print("\n生成测试集预测...")

        # 预测
        test_pred_proba = self.final_model.predict_proba(self.X_test)[:, 1]

        # 分析预测结果
        print("测试集预测分析:")
        print(f"  预测概率范围: {test_pred_proba.min():.6f} - {test_pred_proba.max():.6f}")
        print(f"  预测概率均值: {test_pred_proba.mean():.6f}")
        print(f"  预测概率中位数: {np.median(test_pred_proba):.6f}")

        # 检查预测合理性
        expected_positive_rate = self.y_train.mean()
        predicted_positive_rate = np.mean(test_pred_proba > 0.5)

        print(f"  训练集正样本率: {expected_positive_rate * 100:.2f}%")
        print(f"  预测正样本率(>0.5): {predicted_positive_rate * 100:.2f}%")

        # 创建提交文件
        submission = self.test_ids.copy()
        submission['prob'] = test_pred_proba

        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        submission_path = os.path.join(self.results_dir, f'prediction_fixed_{timestamp}.csv')
        submission.to_csv(submission_path, index=False)

        print(f"预测结果已保存: {submission_path}")

        return submission

    def run_fixed_pipeline(self):
        """运行修复后的完整流程"""
        print("=== 运行修复后的模型训练流程 ===")

        # 1. 加载数据（修复特征选择）
        self.load_data_with_proper_features()

        # 2. 创建平衡数据集
        self.create_balanced_datasets()

        # 3. 模型比较
        results = self.quick_model_comparison()

        # 4. 训练最终模型
        self.train_final_model()

        # 5. 生成预测
        submission = self.generate_predictions()

        print(f"\n=== 修复后的训练流程完成 ===")
        print(f"结果保存在: {self.results_dir}")

        return submission


if __name__ == "__main__":
    print("开始修复后的模型训练...")

    trainer = FixedRepeatBuyerTrainer()
    submission = trainer.run_fixed_pipeline()

    print("修复后的模型训练完成！")