import numpy as np
from tokenization_qwen import QWenTokenizer
from audio import AudioEncoder
import torch
import os
from tqdm import tqdm
import random
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

torch.manual_seed(1234)


class QwenEmbedding:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = QWenTokenizer(vocab_file='qwen.tiktoken')
        self.model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/pythonproject/Embedding/Qwen-Audio-Chat",
                                                          device_map="cuda", trust_remote_code=True).eval()
        self.audio_encoder = self.model.transformer.audio
        self.audio_encoder.eval()

        # 检查model的所有方法
        methods = [method for method in dir(self.model) if not method.startswith('_') and callable(getattr(self.model, method))]
        print("模型的方法:", methods)

    def Qwenencoder(self, audio_path, selected_layers=None):
        layer_outputs = {}

        # 定义前向钩子函数
        def hook_fn(layer_idx):
            def hook(module, input, output):
                layer_outputs[layer_idx] = output

            return hook

        # 注册钩子
        hooks = []
        for layer_idx in selected_layers:
            if 0 <= layer_idx < len(self.audio_encoder.blocks):
                hook = self.audio_encoder.blocks[layer_idx].register_forward_hook(hook_fn(layer_idx))
                hooks.append(hook)
            else:
                print(f"警告: 层索引 {layer_idx} 超出范围 (0-31)")

        # 读取并处理音频
        with torch.no_grad():
            audio_info = self.tokenizer.process_audio(audio_path)
            audios = audio_info["input_audios"].to(self.device)
            audio_span_tokens = audio_info["audio_span_tokens"]
            input_audio_lengths = audio_info["input_audio_lengths"].to(self.device)
            _ = self.audio_encoder.encode(
                input_audios=audios,
                input_audio_lengths=input_audio_lengths,
                audio_span_tokens=audio_span_tokens
            )  # shape: [89, 4096] for a single audio

            # 移除钩子
            for hook in hooks:
                hook.remove()

            # 检查是否获取到了层输出
            if not layer_outputs:
                print("警告: 未捕获到任何层的输出")
                return None

            # 处理获取到的层输出
            features = []
            # 按照层索引顺序处理输出
            for layer_idx in sorted(layer_outputs.keys()):
                layer_output = layer_outputs[layer_idx]

                # 转换为float32以避免bfloat16兼容性问题
                layer_output = layer_output.to(torch.float32)

                # 打印层输出形状
                # print(f"层 {layer_idx} 输出形状: {layer_output.shape}")

                if layer_output.dim() >= 3:
                    avg_feature = layer_output.mean(dim=1)
                    features.append(avg_feature)
                else:
                    features.append(layer_output)

            if len(features) == 1:
                final = features[0]
            else:
                final = torch.stack(features).mean(dim=0)
                print(final.shape)
        return final.cpu().numpy()

    def get_embedding_path(self, audio_dir, selected_layers=None):

        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        embeddings = []
        file_paths = []

        for audio_file in tqdm(audio_files, desc='Processing Audio Files', unit='file'):
            audio_path = os.path.join(audio_dir,
                                      audio_file)  # audio_Path='/root/autodl-tmp/F5-TTS/src/f5_tts/infer/malicious_1/malicious_1.wav'
            file_paths.append(audio_path)
            dict = [
                {'audio': audio_path}
            ]
            audio_path = self.tokenizer.from_list_format(
                dict)  # audio_Path='Audio1:<audio>/root/autodl-tmp/F5-TTS/src/f5_tts/infer/malicious_1/malicious_1.wav</audio>\n'
            embedding = self.Qwenencoder(audio_path, selected_layers=selected_layers)
            embeddings.append(embedding)

        return np.vstack(embeddings)

    def get_embedding_url(self, audio_urls):

        embeddings = []
        for audio_url in audio_urls:
            dict = [
                {"audio": audio_url}
            ]
            audio_url = self.tokenizer.from_list_format(dict)
            embedding = self.Qwenencoder(audio_url)
            embeddings.append(embedding)
        return np.vstack(embeddings)

if __name__ == '__main__':
    # 固定随机种子，确保采样一致
    seed = 42

    malicious_audio_dir = '/home/ubuntu/pythonproject/F5-TTS/tests-malicious'
    benign_audio_dir = '/home/ubuntu/pythonproject/F5-TTS/tests-benign'
    QwenEmbedding = QwenEmbedding()

    # 修改嵌入方法以返回文件名和嵌入向量的映射关系
    embedding_1 = QwenEmbedding.get_embedding_path(malicious_audio_dir, selected_layers=[23,24,25,26])  # shape [N1, 4096]
    embedding_2 = QwenEmbedding.get_embedding_path(benign_audio_dir, selected_layers=[23,24,25,26])  # shape [N2, 4096]
    print(embedding_1[0])

    print(embedding_1.shape, embedding_2.shape)

    # 合并嵌入向量，并记录恶意样本数量
    all_embeddings = np.vstack([embedding_1, embedding_2])
    malicious_count = len(embedding_1)

    # 先用PCA降维以加速t-SNE计算
    n_components_pca = min(50, all_embeddings.shape[1])
    pca = PCA(n_components=n_components_pca)
    pca_embeddings = pca.fit_transform(all_embeddings)
    print(f"PCA降维到{n_components_pca}维, 解释方差比例累计: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 使用t-SNE降维到2维
    # 调整perplexity参数 - 一般在5到50之间效果较好
    # 对于较小的数据集(如<500样本)，较小的perplexity值(5-15)效果较好
    # 对于较大的数据集，较大的perplexity值(30-50)效果较好
    perplexity_value = min(30, all_embeddings.shape[0] // 5)  # 动态设置perplexity

    tsne = TSNE(n_components=2,
                random_state=42,
                perplexity=perplexity_value,
                n_iter=2000,  # 增加迭代次数以获得更稳定结果
                learning_rate='auto',
                init='pca')  # PCA初始化可以提高结果稳定性
    tsne_results = tsne.fit_transform(pca_embeddings)

    # 计算两个类别在t-SNE空间下的中心点
    malicious_center = np.mean(tsne_results[:malicious_count, :], axis=0)
    benign_center = np.mean(tsne_results[malicious_count:, :], axis=0)

    # 为决策边界拟合分类器：恶意设为0，良性设为1
    labels = np.array([0] * malicious_count + [1] * (tsne_results.shape[0] - malicious_count))

    # 使用非线性分类器(SVM)替代逻辑回归
    from sklearn.svm import SVC

    clf = SVC(kernel='rbf', probability=True, gamma='scale').fit(tsne_results, labels)

    # 构造网格数据，用于绘制决策边界
    margin = 2  # 增加边界余量
    x_min, x_max = tsne_results[:, 0].min() - margin, tsne_results[:, 0].max() + margin
    y_min, y_max = tsne_results[:, 1].min() - margin, tsne_results[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # 预测网格上每个点属于良性类别的概率
    Z = clf.predict_proba(grid_points)[:, 1]
    Z = Z.reshape(xx.shape)

    # 计算分类精度
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = clf.predict(tsne_results)
    accuracy = accuracy_score(labels, y_pred)
    report = classification_report(labels, y_pred, target_names=['Malicious', 'Benign'])
    print(f"分类精度: {accuracy:.4f}")
    print("分类报告:\n", report)

    # 绘制t-SNE散点图
    plt.figure(figsize=(14, 12))

    # 主图：散点图和决策边界
    plt.scatter(tsne_results[:malicious_count, 0], tsne_results[:malicious_count, 1],
                c='red', alpha=0.7, label='Malicious', s=100, edgecolors='black')
    plt.scatter(tsne_results[malicious_count:, 0], tsne_results[malicious_count:, 1],
                c='blue', alpha=0.7, label='Benign', s=100, edgecolors='black')

    # 绘制中心点
    plt.scatter(malicious_center[0], malicious_center[1],
                marker='X', s=300, c='darkred', label='Malicious Center', edgecolors='black', linewidths=2)
    plt.scatter(benign_center[0], benign_center[1],
                marker='X', s=300, c='darkblue', label='Benign Center', edgecolors='black', linewidths=2)

    # 绘制决策边界
    plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys_r", linestyles='--', linewidths=3)

    # 添加填充颜色来表示决策区域
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#FFCCCC', '#CCCCFF'], alpha=0.3)

    # 查找错误分类的样本
    misclassified = np.where(y_pred != labels)[0]
    if len(misclassified) > 0:
        plt.scatter(tsne_results[misclassified, 0], tsne_results[misclassified, 1],
                    s=250, facecolors='none', edgecolors='yellow', linewidths=2,
                    label=f'Misclassified ({len(misclassified)})')

    # 添加样本数量和分类精度信息
    plt.annotate(f'Malicious Samples: {malicious_count}',
                 xy=(0.02, 0.98), xycoords='axes fraction', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.annotate(f'Benign Samples: {len(embedding_2)}',
                 xy=(0.02, 0.93), xycoords='axes fraction', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.annotate(f'Classification Accuracy: {accuracy:.2%}',
                 xy=(0.02, 0.88), xycoords='axes fraction', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # 计算类间距离与类内距离
    from scipy.spatial.distance import pdist, squareform

    # 计算类内距离
    malicious_distances = pdist(tsne_results[:malicious_count])
    benign_distances = pdist(tsne_results[malicious_count:])
    avg_malicious_dist = np.mean(malicious_distances)
    avg_benign_dist = np.mean(benign_distances)

    # 计算类间距离
    between_dist = np.linalg.norm(malicious_center - benign_center)

    # 添加距离信息
    plt.annotate(f'Between-Class Distance: {between_dist:.2f}',
                 xy=(0.02, 0.83), xycoords='axes fraction', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.annotate(f'Avg Within-Class Distance: {(avg_malicious_dist + avg_benign_dist) / 2:.2f}',
                 xy=(0.02, 0.78), xycoords='axes fraction', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.legend(fontsize=14, loc='lower right')
    plt.title("t-SNE Visualization of Audio Embeddings with SVM Decision Boundary", fontsize=18)
    plt.xlabel("t-SNE Dimension 1", fontsize=16)
    plt.ylabel("t-SNE Dimension 2", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 替换 plt.show()
    plt.savefig('/home/ubuntu/pythonproject/QwenAudio-1/tsne_analysis.png', dpi=300, bbox_inches='tight')
    print("t-SNE分析图像已保存到 /home/ubuntu/pythonproject/QwenAudio-1/tsne_analysis.png")

    # 可选：添加3D可视化
    from mpl_toolkits.mplot3d import Axes3D

    # 使用t-SNE降维到3维以获得更多信息
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity_value, init='pca')
    tsne_results_3d = tsne_3d.fit_transform(pca_embeddings)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D散点图
    ax.scatter(tsne_results_3d[:malicious_count, 0],
               tsne_results_3d[:malicious_count, 1],
               tsne_results_3d[:malicious_count, 2],
               c='red', alpha=0.7, label='Malicious', s=80)
    ax.scatter(tsne_results_3d[malicious_count:, 0],
               tsne_results_3d[malicious_count:, 1],
               tsne_results_3d[malicious_count:, 2],
               c='blue', alpha=0.7, label='Benign', s=80)

    ax.set_title("3D t-SNE Visualization of Audio Embeddings", fontsize=18)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax.set_zlabel("t-SNE Dimension 3", fontsize=14)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig('/home/ubuntu/pythonproject/QwenAudio-1/tsne_3d_analysis.png', dpi=300, bbox_inches='tight')
    print("3D t-SNE分析图像已保存到 /home/ubuntu/pythonproject/QwenAudio-1/tsne_3d_analysis.png")

