import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
import json
import os
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

# 尝试导入额外的库
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    print("⚠️ 未安装 XGBoost: pip install xgboost")
    HAS_XGB = False

try:
    from sklearn.inspection import permutation_importance

    HAS_PERM_IMP = True
except ImportError:
    print("⚠️ sklearn版本较低，无法使用permutation_importance")
    HAS_PERM_IMP = False

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FeatureImportanceAnalyzer:
    """特征重要性分析器"""

    def __init__(self, feature_dir='../outputs/features', output_dir='../outputs'):
        """
        初始化分析器

        Args:
            feature_dir: 特征文件目录
            output_dir: 输出目录
        """
        self.feature_dir = feature_dir
        self.output_dir = output_dir

        # 创建输出目录
        self.analysis_dir = os.path.join(output_dir, 'feature_analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(os.path.join(self.analysis_dir, 'plots'), exist_ok=True)

        # 数据容器
        self.train_features = None
        self.test_features = None
        self.X_train = None
        self.y_train = None
        self.feature_names = None

        # 结果容器
        self.importance_results = {}

        print(f"✅ 特征重要性分析器初始化完成")
        print(f"📁 输出目录: {self.analysis_dir}")

    def load_features(self, timestamp=None):
        """加载特征文件"""
        print("📊 加载特征文件...")

        # 如果没有指定时间戳，找最新的文件
        if timestamp is None:
            feature_files = [f for f in os.listdir(self.feature_dir) if
                             f.startswith('train_features_') and f.endswith('.csv')]
            if not feature_files:
                raise FileNotFoundError("未找到特征文件")

            # 找最新的文件
            latest_file = sorted(feature_files)[-1]
            timestamp = latest_file.replace('train_features_', '').replace('.csv', '')
            print(f"🔍 自动选择最新特征文件: {timestamp}")

        # 加载特征文件
        train_path = os.path.join(self.feature_dir, f'train_features_{timestamp}.csv')
        test_path = os.path.join(self.feature_dir, f'test_features_{timestamp}.csv')
        info_path = os.path.join(self.feature_dir, f'feature_info_{timestamp}.json')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练特征文件不存在: {train_path}")

        # 加载训练数据
        self.train_features = pd.read_csv(train_path)
        print(f"✅ 训练特征: {self.train_features.shape}")

        # 加载测试数据
        if os.path.exists(test_path):
            self.test_features = pd.read_csv(test_path)
            print(f"✅ 测试特征: {self.test_features.shape}")

        # 加载特征信息
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                feature_info = json.load(f)
                self.feature_names = feature_info.get('feature_names', [])
                print(f"✅ 特征信息: {len(self.feature_names)} 个特征")

        # 准备训练数据
        self._prepare_training_data()

    def _prepare_training_data(self):
        """准备训练数据"""
        print("🔧 准备训练数据...")

        # 需要加载原始训练数据来获取标签
        train_original = pd.read_csv('../data/data_format1/train_format1.csv')

        # 合并特征和标签
        train_data = self.train_features.merge(
            train_original[['user_id', 'merchant_id', 'label']],
            on=['user_id', 'merchant_id'],
            how='left'
        )

        # 准备X和y
        feature_cols = [col for col in train_data.columns
                        if col not in ['user_id', 'merchant_id', 'label']]

        self.X_train = train_data[feature_cols]
        self.y_train = train_data['label']

        if self.feature_names is None:
            self.feature_names = feature_cols

        print(f"✅ 训练数据准备完成: X{self.X_train.shape}, y{self.y_train.shape}")
        print(f"📊 正样本比例: {self.y_train.mean() * 100:.2f}%")

    def correlation_analysis(self):
        """相关性分析"""
        print("\n📈 开始相关性分析...")

        # 计算与目标变量的相关性
        correlations = {}

        for col in self.X_train.columns:
            try:
                # Pearson相关系数
                pearson_corr, pearson_p = pearsonr(self.X_train[col], self.y_train)

                # Spearman相关系数
                spearman_corr, spearman_p = spearmanr(self.X_train[col], self.y_train)

                correlations[col] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'abs_pearson': abs(pearson_corr),
                    'abs_spearman': abs(spearman_corr)
                }
            except Exception as e:
                print(f"  ⚠️ 计算 {col} 相关性失败: {e}")
                correlations[col] = {
                    'pearson_corr': 0, 'pearson_p_value': 1,
                    'spearman_corr': 0, 'spearman_p_value': 1,
                    'abs_pearson': 0, 'abs_spearman': 0
                }

        # 转换为DataFrame并排序
        corr_df = pd.DataFrame(correlations).T
        corr_df = corr_df.sort_values('abs_pearson', ascending=False)

        self.importance_results['correlation'] = corr_df

        # 显示TOP特征
        print(f"\n🎯 相关性分析结果 (TOP 10):")
        for i, (feature, data) in enumerate(corr_df.head(10).iterrows()):
            print(
                f"  {i + 1:2d}. {feature:30s} | Pearson: {data['pearson_corr']:6.3f} | Spearman: {data['spearman_corr']:6.3f}")

        # 可视化
        self._plot_correlation_analysis(corr_df)

        return corr_df

    def _plot_correlation_analysis(self, corr_df):
        """可视化相关性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # TOP 15相关性特征
        top_features = corr_df.head(15)

        # Pearson相关性
        axes[0, 0].barh(range(len(top_features)), top_features['pearson_corr'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features.index, fontsize=8)
        axes[0, 0].set_title('TOP 15 Pearson相关性')
        axes[0, 0].set_xlabel('相关系数')

        # Spearman相关性
        axes[0, 1].barh(range(len(top_features)), top_features['spearman_corr'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features.index, fontsize=8)
        axes[0, 1].set_title('TOP 15 Spearman相关性')
        axes[0, 1].set_xlabel('相关系数')

        # 绝对相关性对比
        axes[1, 0].scatter(top_features['abs_pearson'], top_features['abs_spearman'], alpha=0.7)
        axes[1, 0].plot([0, top_features['abs_pearson'].max()], [0, top_features['abs_pearson'].max()], 'r--',
                        alpha=0.5)
        axes[1, 0].set_xlabel('绝对Pearson相关性')
        axes[1, 0].set_ylabel('绝对Spearman相关性')
        axes[1, 0].set_title('线性 vs 单调相关性')

        # 相关性分布
        axes[1, 1].hist(corr_df['pearson_corr'], bins=30, alpha=0.7, label='Pearson')
        axes[1, 1].hist(corr_df['spearman_corr'], bins=30, alpha=0.7, label='Spearman')
        axes[1, 1].set_xlabel('相关系数')
        axes[1, 1].set_ylabel('特征数量')
        axes[1, 1].set_title('相关性分布')
        axes[1, 1].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.analysis_dir, 'plots', 'correlation_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 相关性分析图已保存: {plot_path}")

    def statistical_feature_selection(self):
        """统计特征选择"""
        print("\n📊 开始统计特征选择...")

        results = {}

        # 1. 卡方检验 (适用于非负特征)
        try:
            # 确保特征非负
            X_non_negative = self.X_train.copy()
            for col in X_non_negative.columns:
                if X_non_negative[col].min() < 0:
                    X_non_negative[col] = X_non_negative[col] - X_non_negative[col].min()

            chi2_selector = SelectKBest(chi2, k='all')
            chi2_selector.fit(X_non_negative, self.y_train)

            chi2_scores = pd.DataFrame({
                'feature': self.X_train.columns,
                'chi2_score': chi2_selector.scores_,
                'chi2_p_value': chi2_selector.pvalues_
            }).sort_values('chi2_score', ascending=False)

            results['chi2'] = chi2_scores
            print("✅ 卡方检验完成")

        except Exception as e:
            print(f"⚠️ 卡方检验失败: {e}")

        # 2. F检验
        try:
            f_selector = SelectKBest(f_classif, k='all')
            f_selector.fit(self.X_train, self.y_train)

            f_scores = pd.DataFrame({
                'feature': self.X_train.columns,
                'f_score': f_selector.scores_,
                'f_p_value': f_selector.pvalues_
            }).sort_values('f_score', ascending=False)

            results['f_test'] = f_scores
            print("✅ F检验完成")

        except Exception as e:
            print(f"⚠️ F检验失败: {e}")

        self.importance_results['statistical'] = results

        # 显示结果
        if 'f_test' in results:
            print(f"\n🎯 F检验结果 (TOP 10):")
            for i, (_, row) in enumerate(results['f_test'].head(10).iterrows()):
                print(
                    f"  {i + 1:2d}. {row['feature']:30s} | F-score: {row['f_score']:8.2f} | p-value: {row['f_p_value']:.2e}")

        return results

    def model_based_importance(self):
        """基于模型的特征重要性"""
        print("\n🤖 开始基于模型的特征重要性分析...")

        results = {}

        # 1. Random Forest
        print("  🌳 Random Forest...")
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',  # 处理类别不平衡
                n_jobs=-1
            )
            rf.fit(self.X_train, self.y_train)

            rf_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            results['random_forest'] = rf_importance
            print("    ✅ Random Forest完成")

        except Exception as e:
            print(f"    ❌ Random Forest失败: {e}")

        # 2. LightGBM
        print("  💡 LightGBM...")
        try:
            # 创建LightGBM数据集
            train_data = lgb.Dataset(self.X_train, label=self.y_train)

            # 参数设置
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'is_unbalance': True  # 处理类别不平衡
            }

            # 训练模型
            lgb_model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

            # 获取特征重要性
            lgb_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': lgb_model.feature_importance(importance_type='gain'),
                'split_importance': lgb_model.feature_importance(importance_type='split')
            }).sort_values('importance', ascending=False)

            results['lightgbm'] = lgb_importance
            print("    ✅ LightGBM完成")

        except Exception as e:
            print(f"    ❌ LightGBM失败: {e}")

        # 3. XGBoost (如果可用)
        if HAS_XGB:
            print("  🚀 XGBoost...")
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    scale_pos_weight=len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1]),
                    # 处理不平衡
                    n_jobs=-1
                )
                xgb_model.fit(self.X_train, self.y_train)

                xgb_importance = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': xgb_model.feature_importances_
                }).sort_values('importance', ascending=False)

                results['xgboost'] = xgb_importance
                print("    ✅ XGBoost完成")

            except Exception as e:
                print(f"    ❌ XGBoost失败: {e}")

        # 4. 递归特征消除 (RFE)
        print("  🔄 递归特征消除...")
        try:
            # 使用逻辑回归作为基础估计器
            lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

            # 选择TOP 30特征
            rfe = RFE(estimator=lr, n_features_to_select=30, step=1)
            rfe.fit(self.X_train, self.y_train)

            rfe_results = pd.DataFrame({
                'feature': self.X_train.columns,
                'selected': rfe.support_,
                'ranking': rfe.ranking_
            }).sort_values('ranking')

            results['rfe'] = rfe_results
            print("    ✅ 递归特征消除完成")

        except Exception as e:
            print(f"    ❌ 递归特征消除失败: {e}")

        self.importance_results['model_based'] = results

        # 显示结果
        if 'lightgbm' in results:
            print(f"\n🎯 LightGBM特征重要性 (TOP 10):")
            for i, (_, row) in enumerate(results['lightgbm'].head(10).iterrows()):
                print(f"  {i + 1:2d}. {row['feature']:30s} | 重要性: {row['importance']:8.2f}")

        # 可视化
        self._plot_model_importance(results)

        return results

    def _plot_model_importance(self, results):
        """可视化模型重要性"""
        n_models = len(results)
        if n_models == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        plot_idx = 0

        for model_name, importance_df in results.items():
            if plot_idx >= 4:
                break

            if model_name == 'rfe':
                # RFE结果特殊处理
                selected_features = importance_df[importance_df['selected']].head(15)
                axes[plot_idx].barh(range(len(selected_features)), [1] * len(selected_features))
                axes[plot_idx].set_yticks(range(len(selected_features)))
                axes[plot_idx].set_yticklabels(selected_features['feature'], fontsize=8)
                axes[plot_idx].set_title(f'{model_name.upper()} 选中特征')
            else:
                # 其他模型的重要性
                top_features = importance_df.head(15)
                importance_col = 'importance' if 'importance' in importance_df.columns else importance_df.columns[1]

                axes[plot_idx].barh(range(len(top_features)), top_features[importance_col])
                axes[plot_idx].set_yticks(range(len(top_features)))
                axes[plot_idx].set_yticklabels(top_features['feature'], fontsize=8)
                axes[plot_idx].set_title(f'{model_name.upper()} 特征重要性')
                axes[plot_idx].set_xlabel('重要性得分')

            plot_idx += 1

        # 隐藏未使用的子图
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plot_path = os.path.join(self.analysis_dir, 'plots', 'model_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 模型重要性图已保存: {plot_path}")

    def permutation_importance_analysis(self):
        """排列重要性分析"""
        if not HAS_PERM_IMP:
            print("⚠️ sklearn版本不支持permutation_importance，跳过")
            return None

        print("\n🔀 开始排列重要性分析...")

        try:
            # 使用LightGBM模型
            lgb_model = lgb.LGBMClassifier(
                n_estimators=50,  # 减少树的数量以加速
                random_state=42,
                is_unbalance=True,
                verbose=-1
            )
            lgb_model.fit(self.X_train, self.y_train)

            # 计算排列重要性
            perm_importance = permutation_importance(
                lgb_model, self.X_train, self.y_train,
                n_repeats=5,  # 重复次数
                random_state=42,
                scoring='roc_auc'
            )

            # 整理结果
            perm_results = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)

            self.importance_results['permutation'] = perm_results

            print(f"🎯 排列重要性分析结果 (TOP 10):")
            for i, (_, row) in enumerate(perm_results.head(10).iterrows()):
                print(
                    f"  {i + 1:2d}. {row['feature']:30s} | 重要性: {row['importance_mean']:6.4f} ± {row['importance_std']:6.4f}")

            return perm_results

        except Exception as e:
            print(f"❌ 排列重要性分析失败: {e}")
            return None

    def feature_stability_analysis(self):
        """特征稳定性分析"""
        print("\n🔄 开始特征稳定性分析...")

        try:
            # 使用交叉验证评估特征稳定性
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            stability_results = {}

            for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
                print(f"  📊 处理第 {fold + 1} 折...")

                X_fold = self.X_train.iloc[train_idx]
                y_fold = self.y_train.iloc[train_idx]

                # 训练LightGBM
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=50,
                    random_state=42,
                    is_unbalance=True,
                    verbose=-1
                )
                lgb_model.fit(X_fold, y_fold)

                # 记录特征重要性
                for i, feature in enumerate(self.X_train.columns):
                    if feature not in stability_results:
                        stability_results[feature] = []
                    stability_results[feature].append(lgb_model.feature_importances_[i])

            # 计算稳定性指标
            stability_df = pd.DataFrame({
                'feature': list(stability_results.keys()),
                'mean_importance': [np.mean(scores) for scores in stability_results.values()],
                'std_importance': [np.std(scores) for scores in stability_results.values()],
                'cv_importance': [np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else float('inf')
                                  for scores in stability_results.values()]
            })

            # 按平均重要性排序
            stability_df = stability_df.sort_values('mean_importance', ascending=False)

            self.importance_results['stability'] = stability_df

            print(f"🎯 特征稳定性分析结果 (TOP 10):")
            for i, (_, row) in enumerate(stability_df.head(10).iterrows()):
                print(
                    f"  {i + 1:2d}. {row['feature']:30s} | 平均: {row['mean_importance']:6.4f} | CV: {row['cv_importance']:6.4f}")

            return stability_df

        except Exception as e:
            print(f"❌ 特征稳定性分析失败: {e}")
            return None

    def comprehensive_feature_ranking(self):
        """综合特征排名"""
        print("\n🏆 开始综合特征排名...")

        # 收集所有可用的重要性结果
        ranking_data = {}

        # 初始化特征列表
        for feature in self.X_train.columns:
            ranking_data[feature] = {'scores': [], 'methods': []}

        # 1. 相关性分析
        if 'correlation' in self.importance_results:
            corr_df = self.importance_results['correlation']
            for feature in corr_df.index:
                if feature in ranking_data:
                    # 使用绝对相关性作为得分
                    score = corr_df.loc[feature, 'abs_pearson']
                    ranking_data[feature]['scores'].append(score)
                    ranking_data[feature]['methods'].append('correlation')

        # 2. 模型重要性
        if 'model_based' in self.importance_results:
            for model_name, importance_df in self.importance_results['model_based'].items():
                if model_name == 'rfe':
                    # RFE特殊处理：选中的特征得分为1，未选中为0
                    for _, row in importance_df.iterrows():
                        feature = row['feature']
                        if feature in ranking_data:
                            score = 1.0 if row['selected'] else 0.0
                            ranking_data[feature]['scores'].append(score)
                            ranking_data[feature]['methods'].append(f'rfe')
                else:
                    # 标准化重要性得分
                    importance_col = 'importance' if 'importance' in importance_df.columns else importance_df.columns[1]
                    max_importance = importance_df[importance_col].max()

                    for _, row in importance_df.iterrows():
                        feature = row['feature']
                        if feature in ranking_data and max_importance > 0:
                            score = row[importance_col] / max_importance
                            ranking_data[feature]['scores'].append(score)
                            ranking_data[feature]['methods'].append(f'model_{model_name}')

        # 3. 排列重要性
        if 'permutation' in self.importance_results:
            perm_df = self.importance_results['permutation']
            max_perm = perm_df['importance_mean'].max()
            if max_perm > 0:
                for _, row in perm_df.iterrows():
                    feature = row['feature']
                    if feature in ranking_data:
                        score = row['importance_mean'] / max_perm
                        ranking_data[feature]['scores'].append(score)
                        ranking_data[feature]['methods'].append('permutation')

        # 4. 稳定性分析
        if 'stability' in self.importance_results:
            stability_df = self.importance_results['stability']
            max_stability = stability_df['mean_importance'].max()
            if max_stability > 0:
                for _, row in stability_df.iterrows():
                    feature = row['feature']
                    if feature in ranking_data:
                        score = row['mean_importance'] / max_stability
                        ranking_data[feature]['scores'].append(score)
                        ranking_data[feature]['methods'].append('stability')

        # 计算综合得分
        comprehensive_results = []
        for feature, data in ranking_data.items():
            if data['scores']:
                mean_score = np.mean(data['scores'])
                std_score = np.std(data['scores']) if len(data['scores']) > 1 else 0
                method_count = len(data['scores'])

                comprehensive_results.append({
                    'feature': feature,
                    '综合得分': mean_score,
                    '得分标准差': std_score,
                    '方法数量': method_count,
                    '稳定性': 1 - (std_score / mean_score if mean_score > 0 else 1),
                    '方法列表': ', '.join(data['methods'])
                })

        # 转换为DataFrame并排序
        final_ranking = pd.DataFrame(comprehensive_results)
        final_ranking = final_ranking.sort_values('综合得分', ascending=False)

        self.importance_results['comprehensive'] = final_ranking

        # 显示TOP特征
        print(f"🎯 综合特征排名 (TOP 15):")
        print(f"{'排名':>4} {'特征名':30} {'综合得分':>8} {'稳定性':>8} {'方法数':>6}")
        print("-" * 70)

        for i, (_, row) in enumerate(final_ranking.head(15).iterrows()):
            print(f"{i + 1:4d} {row['feature']:30} {row['综合得分']:8.4f} {row['稳定性']:8.4f} {row['方法数量']:6d}")

        # 可视化综合排名
        self._plot_comprehensive_ranking(final_ranking)

        return final_ranking

    def _plot_comprehensive_ranking(self, ranking_df):
        """可视化综合排名"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # TOP 15特征综合得分
        top_15 = ranking_df.head(15)

        axes[0, 0].barh(range(len(top_15)), top_15['综合得分'])
        axes[0, 0].set_yticks(range(len(top_15)))
        axes[0, 0].set_yticklabels(top_15['feature'], fontsize=8)
        axes[0, 0].set_title('TOP 15 综合特征重要性')
        axes[0, 0].set_xlabel('综合得分')

        # 稳定性 vs 重要性散点图
        axes[0, 1].scatter(ranking_df['综合得分'], ranking_df['稳定性'], alpha=0.6)
        axes[0, 1].set_xlabel('综合得分')
        axes[0, 1].set_ylabel('稳定性')
        axes[0, 1].set_title('重要性 vs 稳定性')

        # 方法数量分布
        method_counts = ranking_df['方法数量'].value_counts().sort_index()
        axes[1, 0].bar(method_counts.index, method_counts.values)
        axes[1, 0].set_xlabel('使用的方法数量')
        axes[1, 0].set_ylabel('特征数量')
        axes[1, 0].set_title('特征评估方法数量分布')

        # 综合得分分布
        axes[1, 1].hist(ranking_df['综合得分'], bins=20, alpha=0.7)
        axes[1, 1].set_xlabel('综合得分')
        axes[1, 1].set_ylabel('特征数量')
        axes[1, 1].set_title('综合得分分布')

        plt.tight_layout()
        plot_path = os.path.join(self.analysis_dir, 'plots', 'comprehensive_ranking.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 综合排名图已保存: {plot_path}")

    def feature_selection_recommendation(self, top_k=30):
        """特征选择建议"""
        print(f"\n💡 特征选择建议 (推荐TOP {top_k}特征)...")

        if 'comprehensive' in self.importance_results:
            comprehensive_ranking = self.importance_results['comprehensive']

            # 基于综合得分选择TOP特征
            recommended_features = comprehensive_ranking.head(top_k)['feature'].tolist()

            # 特征类型分析
            feature_types = {
                'user_features': [f for f in recommended_features if f.startswith('user_')],
                'merchant_features': [f for f in recommended_features if f.startswith('merchant_')],
                'behavior_features': [f for f in recommended_features if
                                      any(x in f for x in ['action_', 'total_actions', 'unique_'])],
                'cross_features': [f for f in recommended_features if '_x_' in f or 'relative' in f],
                'other_features': [f for f in recommended_features if not any(
                    x in f for x in ['user_', 'merchant_', 'action_', 'total_actions', 'unique_', '_x_', 'relative'])]
            }

            print(f"📊 推荐特征分布:")
            for feature_type, features in feature_types.items():
                if features:
                    print(f"  {feature_type:20}: {len(features):2d}个")

            print(f"\n🎯 TOP {top_k} 推荐特征:")
            for i, feature in enumerate(recommended_features):
                score = comprehensive_ranking[comprehensive_ranking['feature'] == feature]['综合得分'].iloc[0]
                print(f"  {i + 1:2d}. {feature:35} (得分: {score:.4f})")

            # 保存推荐列表
            recommendation = {
                'timestamp': datetime.now().isoformat(),
                'top_k': top_k,
                'recommended_features': recommended_features,
                'feature_type_distribution': {k: len(v) for k, v in feature_types.items()},
                'comprehensive_scores': comprehensive_ranking.head(top_k).to_dict('records')
            }

            rec_path = os.path.join(self.analysis_dir, f'feature_recommendation_top{top_k}.json')
            with open(rec_path, 'w', encoding='utf-8') as f:
                json.dump(recommendation, f, ensure_ascii=False, indent=2)

            print(f"💾 特征推荐已保存: {rec_path}")

            return recommended_features
        else:
            print("⚠️ 需要先运行综合特征排名分析")
            return None

    def generate_analysis_report(self):
        """生成分析报告"""
        print("\n📋 生成特征重要性分析报告...")

        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'train_samples': len(self.X_train),
                'feature_count': len(self.X_train.columns),
                'positive_rate': float(self.y_train.mean()),
                'feature_names': list(self.X_train.columns)
            },
            'analysis_methods': list(self.importance_results.keys()),
            'top_features_by_method': {}
        }

        # 收集各方法的TOP特征
        for method, results in self.importance_results.items():
            if method == 'correlation':
                top_features = results.head(10).index.tolist()
                report['top_features_by_method'][method] = top_features
            elif method == 'model_based':
                for model_name, model_results in results.items():
                    if model_name != 'rfe':
                        importance_col = 'importance' if 'importance' in model_results.columns else \
                        model_results.columns[1]
                        top_features = model_results.head(10)['feature'].tolist()
                        report['top_features_by_method'][f'{method}_{model_name}'] = top_features
            elif method in ['permutation', 'stability', 'comprehensive']:
                if isinstance(results, pd.DataFrame) and 'feature' in results.columns:
                    top_features = results.head(10)['feature'].tolist()
                    report['top_features_by_method'][method] = top_features

        # 保存报告
        report_path = os.path.join(self.analysis_dir, 'feature_importance_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✅ 分析报告已保存: {report_path}")

        # 生成README
        self._generate_readme()

        return report

    def _generate_readme(self):
        """生成README文件"""
        readme_content = f"""# 特征重要性分析报告

## 分析概述
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据规模**: {len(self.X_train):,} 样本 × {len(self.X_train.columns)} 特征
- **正样本比例**: {self.y_train.mean() * 100:.2f}%

## 分析方法
"""

        if 'correlation' in self.importance_results:
            readme_content += "✅ 相关性分析 (Pearson & Spearman)\n"
        if 'statistical' in self.importance_results:
            readme_content += "✅ 统计检验 (卡方检验 & F检验)\n"
        if 'model_based' in self.importance_results:
            readme_content += "✅ 模型重要性 (Random Forest, LightGBM, XGBoost)\n"
        if 'permutation' in self.importance_results:
            readme_content += "✅ 排列重要性分析\n"
        if 'stability' in self.importance_results:
            readme_content += "✅ 特征稳定性分析\n"
        if 'comprehensive' in self.importance_results:
            readme_content += "✅ 综合特征排名\n"

        readme_content += f"""
## 关键发现

### TOP 10 重要特征
"""

        if 'comprehensive' in self.importance_results:
            top_features = self.importance_results['comprehensive'].head(10)
            for i, (_, row) in enumerate(top_features.iterrows()):
                readme_content += f"{i + 1}. **{row['feature']}** (得分: {row['综合得分']:.4f})\n"

        readme_content += f"""
## 文件说明
- `plots/`: 所有分析图表
- `feature_importance_report.json`: 完整分析结果
- `feature_recommendation_top30.json`: 特征选择建议

## 建议
基于分析结果，建议在模型训练中优先使用TOP 30特征，可以在保持性能的同时降低模型复杂度。

---
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        readme_path = os.path.join(self.analysis_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"📖 README已保存: {readme_path}")

    def run_full_analysis(self, top_k=30):
        """运行完整分析流程"""
        print("🚀 开始完整特征重要性分析...")

        # 1. 相关性分析
        self.correlation_analysis()

        # 2. 统计特征选择
        self.statistical_feature_selection()

        # 3. 模型重要性
        self.model_based_importance()

        # 4. 排列重要性
        self.permutation_importance_analysis()

        # 5. 稳定性分析
        self.feature_stability_analysis()

        # 6. 综合排名
        self.comprehensive_feature_ranking()

        # 7. 特征选择建议
        recommended_features = self.feature_selection_recommendation(top_k)

        # 8. 生成报告
        self.generate_analysis_report()

        print(f"\n🎉 特征重要性分析完成！")
        print(f"📁 结果保存在: {self.analysis_dir}")
        print(f"🎯 推荐使用TOP {top_k}特征进行建模")

        return recommended_features


# 使用示例
if __name__ == "__main__":
    print("🚀 开始特征重要性分析...")

    # 初始化分析器
    analyzer = FeatureImportanceAnalyzer(
        feature_dir='../outputs/features',
        output_dir='../outputs'
    )

    # 加载特征文件
    analyzer.load_features()  # 自动找最新的特征文件

    # 运行完整分析
    recommended_features = analyzer.run_full_analysis(top_k=30)

    print("\n📊 分析总结:")
    print(f"💡 推荐的TOP 30特征可以显著提升模型性能")
    print(f"🎯 建议在模型训练中使用这些特征")
    print(f"📁 查看 {analyzer.analysis_dir} 目录获取详细结果")