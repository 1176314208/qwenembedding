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
    audio_dir = 'your audio path'
    QwenEmbedding = QwenEmbedding()

    embedding= QwenEmbedding.get_embedding_path(audio_dir, selected_layers=[31])  # shape [N1, 4096]

