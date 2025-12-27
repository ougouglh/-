import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os
from datetime import datetime

# 尝试导入 tqdm，如果没有则忽略
try:
    from tqdm import tqdm

    tqdm.pandas()
    HAS_TQDM = True
except ImportError:
    print("⚠️ 未安装 tqdm，将不显示进度条。安装命令: pip install tqdm")
    HAS_TQDM = False

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RepeatBuyerAnalyzer:
    """重复购买预测数据分析类 - 修复版本"""

    def __init__(self, train_path, user_info_path=None, user_log_path=None, output_dir='analysis_results'):
        """
        初始化分析器

        Args:
            train_path: 训练数据路径
            user_info_path: 用户信息数据路径（可选）
            user_log_path: 用户行为日志路径（可选）
            output_dir: 输出结果保存目录
        """
        self.train_path = train_path
        self.user_info_path = user_info_path
        self.user_log_path = user_log_path
        self.output_dir = output_dir

        # 创建输出目录
        self._create_output_dirs()

        # 加载数据
        self.train_df = None
        self.user_info_df = None
        self.user_log_df = None

        self._load_data()

    def _create_output_dirs(self):
        """创建输出目录"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
            print(f"✅ 输出目录已创建: {self.output_dir}")
        except Exception as e:
            print(f"⚠️ 创建输出目录失败: {e}")
            self.output_dir = '.'  # 使用当前目录作为备选

    def _load_data(self):
        """加载所有数据文件"""
        print("=== 开始加载数据 ===")

        # 加载训练数据
        try:
            print("📊 正在加载训练数据...")
            self.train_df = pd.read_csv(self.train_path)
            print(f"✅ 训练数据加载成功: {self.train_df.shape}")

            # 保存基础统计信息
            self._save_basic_stats()

        except Exception as e:
            print(f"❌ 训练数据加载失败: {e}")
            raise e

        # 加载用户信息（可选）
        if self.user_info_path and os.path.exists(self.user_info_path):
            try:
                print("👤 正在加载用户信息...")
                self.user_info_df = pd.read_csv(self.user_info_path)
                print(f"✅ 用户信息加载成功: {self.user_info_df.shape}")
            except Exception as e:
                print(f"⚠️ 用户信息加载失败: {e}")
        elif self.user_info_path:
            print(f"⚠️ 用户信息文件不存在: {self.user_info_path}")

        # 加载用户行为日志（可选）
        if self.user_log_path and os.path.exists(self.user_log_path):
            try:
                print("📱 正在加载用户行为日志...")
                file_size = os.path.getsize(self.user_log_path)

                if file_size > 100 * 1024 * 1024 and HAS_TQDM:  # 大于100MB且有tqdm
                    print("⏳ 检测到大文件，使用分批加载...")
                    chunks = []
                    chunk_iter = pd.read_csv(self.user_log_path, chunksize=50000)
                    for chunk in tqdm(chunk_iter, desc="加载数据块"):
                        chunks.append(chunk)
                    self.user_log_df = pd.concat(chunks, ignore_index=True)
                else:
                    self.user_log_df = pd.read_csv(self.user_log_path)
                print(f"✅ 用户行为日志加载成功: {self.user_log_df.shape}")
            except Exception as e:
                print(f"⚠️ 用户行为日志加载失败: {e}")
        elif self.user_log_path:
            print(f"⚠️ 用户行为日志文件不存在: {self.user_log_path}")

    def _save_basic_stats(self):
        """保存基础统计信息"""
        try:
            stats = {
                'load_time': datetime.now().isoformat(),
                'train_shape': list(self.train_df.shape),
                'train_columns': list(self.train_df.columns),
                'train_dtypes': self.train_df.dtypes.astype(str).to_dict(),
                'train_memory_usage': int(self.train_df.memory_usage(deep=True).sum()),
            }

            stats_path = os.path.join(self.output_dir, 'basic_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"💾 基础统计信息已保存: {stats_path}")
        except Exception as e:
            print(f"⚠️ 保存基础统计信息失败: {e}")

    def _save_plot(self, filename):
        """保存当前图片"""
        try:
            plot_path = os.path.join(self.output_dir, 'plots', f"{filename}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 图片已保存: {plot_path}")
        except Exception as e:
            print(f"⚠️ 保存图片失败: {e}")

    def _save_data(self, data, filename, data_type='json'):
        """保存数据到文件"""
        try:
            if data_type == 'json':
                file_path = os.path.join(self.output_dir, 'data', f"{filename}.json")
                if isinstance(data, pd.Series):
                    data_to_save = data.to_dict()
                elif isinstance(data, pd.DataFrame):
                    data_to_save = data.to_dict('records')
                else:
                    data_to_save = data

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2, default=str)

            elif data_type == 'csv':
                file_path = os.path.join(self.output_dir, 'data', f"{filename}.csv")
                if isinstance(data, (pd.Series, pd.DataFrame)):
                    data.to_csv(file_path, index=True)
                else:
                    pd.DataFrame(data).to_csv(file_path, index=False)

            print(f"💾 数据已保存: {file_path}")
            return file_path
        except Exception as e:
            print(f"⚠️ 保存数据失败: {e}")
            return None

    def basic_info_analysis(self):
        """基础信息分析"""
        print("\n" + "=" * 50)
        print("📊 基础数据分析")
        print("=" * 50)

        df = self.train_df

        # 基本统计
        print("✅ 数据基本信息:")
        print(f"   总样本数: {len(df):,}")
        print(f"   唯一用户数: {df['user_id'].nunique():,}")
        print(f"   唯一商家数: {df['merchant_id'].nunique():,}")
        print(f"   平均每用户关联商家数: {len(df) / df['user_id'].nunique():.2f}")
        print(f"   平均每商家关联用户数: {len(df) / df['merchant_id'].nunique():.2f}")

        # 数据质量检查
        print("\n✅ 数据质量检查:")
        print(f"   缺失值情况:")
        missing_info = df.isnull().sum()
        for col, missing_count in missing_info.items():
            print(f"     {col}: {missing_count} ({missing_count / len(df) * 100:.2f}%)")

        # 重复值检查
        duplicates = df.duplicated().sum()
        print(f"   重复行数: {duplicates}")

        return df.describe()

    def label_distribution_analysis(self):
        """标签分布分析"""
        print("\n" + "=" * 50)
        print("🎯 标签分布分析")
        print("=" * 50)

        df = self.train_df

        # 标签统计
        print("⏳ 正在分析标签分布...")
        label_counts = df['label'].value_counts().sort_index()
        total_labeled = label_counts.sum()

        print("✅ 标签分布:")
        for label, count in label_counts.items():
            percentage = count / total_labeled * 100
            print(f"   标签 {label}: {count:,} 样本 ({percentage:.2f}%)")

        # 正样本比例
        positive_rate = label_counts.get(1, 0) / total_labeled * 100
        print(f"\n🎯 关键指标:")
        print(f"   正样本比例: {positive_rate:.2f}%")
        print(f"   负样本比例: {100 - positive_rate:.2f}%")

        if positive_rate < 20:
            print("   ⚠️ 检测到严重的类别不平衡问题！")
            print("   💡 建议使用: SMOTE、类别权重、阈值优化等方法")

        # 保存分析结果
        label_analysis_result = {
            'label_counts': {str(k): int(v) for k, v in label_counts.items()},
            'positive_rate': float(positive_rate),
            'negative_rate': float(100 - positive_rate),
            'is_imbalanced': positive_rate < 20,
            'analysis_time': datetime.now().isoformat()
        }
        self._save_data(label_analysis_result, 'label_distribution_analysis')

        # 可视化
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        label_counts.plot(kind='bar', color=['#ff7f7f', '#7fbf7f'])
        plt.title('标签分布 (绝对数量)')
        plt.xlabel('标签')
        plt.ylabel('样本数量')
        plt.xticks(rotation=0)

        plt.subplot(1, 2, 2)
        plt.pie(label_counts.values, labels=[f'标签{i}' for i in label_counts.index],
                autopct='%1.1f%%', colors=['#ff7f7f', '#7fbf7f'])
        plt.title('标签分布 (百分比)')

        plt.tight_layout()
        self._save_plot('label_distribution')
        plt.show()

        return label_counts

    def user_behavior_analysis(self):
        """用户行为模式分析"""
        print("\n" + "=" * 50)
        print("👤 用户行为模式分析")
        print("=" * 50)

        df = self.train_df

        # 用户关联商家数分析
        print("⏳ 正在分析用户-商家关联模式...")
        user_merchant_counts = df.groupby('user_id')['merchant_id'].count()

        print("✅ 用户关联商家数分布:")
        merchant_count_dist = user_merchant_counts.value_counts().sort_index()
        total_users = len(user_merchant_counts)

        for count, users in merchant_count_dist.head(10).items():
            percentage = users / total_users * 100
            print(f"   关联{count}个商家的用户: {users:,}个 ({percentage:.2f}%)")

        # 用户忠诚度分析
        print("⏳ 正在计算用户忠诚度指标...")
        single_merchant_users = merchant_count_dist.get(1, 0)
        multi_merchant_users = total_users - single_merchant_users

        print(f"\n🎯 用户忠诚度洞察:")
        print(f"   单一商家用户: {single_merchant_users:,} ({single_merchant_users / total_users * 100:.2f}%)")
        print(f"   多商家用户: {multi_merchant_users:,} ({multi_merchant_users / total_users * 100:.2f}%)")

        # 多商家用户的重复购买率
        print("⏳ 正在分析不同用户群体的重复购买率...")
        multi_users = user_merchant_counts[user_merchant_counts > 1].index
        multi_user_repeat_rate = df[df['user_id'].isin(multi_users)]['label'].mean()
        single_user_repeat_rate = df[~df['user_id'].isin(multi_users)]['label'].mean()

        print(f"   多商家用户重复购买率: {multi_user_repeat_rate * 100:.2f}%")
        print(f"   单商家用户重复购买率: {single_user_repeat_rate * 100:.2f}%")

        # 保存分析结果
        user_behavior_result = {
            'merchant_count_distribution': {str(k): int(v) for k, v in merchant_count_dist.head(20).items()},
            'total_users': int(total_users),
            'single_merchant_users': int(single_merchant_users),
            'multi_merchant_users': int(multi_merchant_users),
            'single_merchant_percentage': float(single_merchant_users / total_users * 100),
            'multi_merchant_percentage': float(multi_merchant_users / total_users * 100),
            'multi_user_repeat_rate': float(multi_user_repeat_rate),
            'single_user_repeat_rate': float(single_user_repeat_rate),
            'analysis_time': datetime.now().isoformat()
        }
        self._save_data(user_behavior_result, 'user_behavior_analysis')

        # 保存详细数据
        self._save_data(user_merchant_counts, 'user_merchant_counts', 'csv')

        # 可视化
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        merchant_count_dist.head(10).plot(kind='bar', color='skyblue')
        plt.title('用户关联商家数分布')
        plt.xlabel('关联商家数')
        plt.ylabel('用户数量')

        plt.subplot(1, 3, 2)
        loyalty_data = [single_merchant_users, multi_merchant_users]
        loyalty_labels = ['单一商家', '多商家']
        plt.pie(loyalty_data, labels=loyalty_labels, autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'])
        plt.title('用户忠诚度分布')

        plt.subplot(1, 3, 3)
        repeat_rates = [single_user_repeat_rate * 100, multi_user_repeat_rate * 100]
        plt.bar(loyalty_labels, repeat_rates, color=['#ff9999', '#66b3ff'])
        plt.title('不同类型用户重复购买率')
        plt.ylabel('重复购买率 (%)')

        plt.tight_layout()
        self._save_plot('user_behavior_analysis')
        plt.show()

        return user_merchant_counts

    def quick_analysis(self):
        """快速分析 - 仅包含核心功能"""
        print("🚀 执行快速分析...")

        # 基础信息
        basic_stats = self.basic_info_analysis()

        # 标签分布
        label_dist = self.label_distribution_analysis()

        # 用户行为
        user_behavior = self.user_behavior_analysis()

        # 生成简要报告
        total_samples = len(self.train_df)
        positive_rate = self.train_df['label'].mean() * 100
        single_merchant_rate = (user_behavior == 1).mean() * 100

        print("\n" + "=" * 60)
        print("🎯 快速分析总结")
        print("=" * 60)
        print(f"📊 数据规模: {total_samples:,} 样本")
        print(f"⚖️ 正样本比例: {positive_rate:.2f}%")
        print(f"👤 单一商家用户比例: {single_merchant_rate:.1f}%")

        if positive_rate < 10:
            print("⚠️ 严重类别不平衡 - 需要特殊处理")

        # 保存简要报告
        quick_report = {
            'total_samples': int(total_samples),
            'positive_rate': float(positive_rate),
            'single_merchant_rate': float(single_merchant_rate),
            'is_imbalanced': positive_rate < 20,
            'analysis_time': datetime.now().isoformat()
        }
        self._save_data(quick_report, 'quick_analysis_report')

        print(f"\n💾 分析结果已保存到: {self.output_dir}")

        return {
            'basic_stats': basic_stats,
            'label_distribution': label_dist,
            'user_behavior': user_behavior,
            'quick_report': quick_report
        }


# 使用示例
if __name__ == "__main__":
    print("🚀 开始数据分析...")

    # 使用你的文件路径
    analyzer = RepeatBuyerAnalyzer(
        train_path='../data/data_format1/train_format1.csv',
        user_info_path='../data/data_format1/user_info_format1.csv',
        user_log_path='../data/data_format1/user_log_format1.csv',
        output_dir='../outputs'
    )

    # 执行快速分析（推荐）
    results = analyzer.quick_analysis()

    print("\n🎉 分析完成！")
    print("📁 请查看 ../outputs 目录获取完整结果")