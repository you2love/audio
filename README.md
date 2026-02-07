# 计算机音频处理学习网站 版

这是一个完整的计算机音频处理学习网站，包含传统算法和年最新的AI音频处理技术。

## 网站结构

```
audio-tutorial/
├── index.html              # 主页面（更新）
├── index.html       # 音频基础知识
├── algorithms.html         # 常用算法详解（更新）
├── ai-audio.html          # AI音频处理（新增）
├── python-examples.html   # Python示例（更新）
├── golang-examples.html   # Golang示例（更新）
└── css/
    └── style.css          # 网站样式
```

## 年内容更新

### 新增AI音频处理章节
- Transformer在音频中的应用
- 扩散模型音频生成
- 神经音频编解码器
- 实时AI语音增强
- 预训练音频模型

### 更新的内容
- 采样率与AI超分辨率
- 神经滤波器设计
- AI增强的FFT处理
- ONNX模型部署
- 端侧AI音频处理

## 内容概览

### 1. 音频基础 (index.html)
- 声音的本质与传播
- 模拟信号与数字信号
- 采样率与位深度（ AI优化标准）
- 时域与频域分析（AI增强）
- 声道与AI空间音频

### 2. 常用算法 (algorithms.html)
- FFT（AI超分辨率增强）
- 数字滤波器（神经网络滤波器）
- 窗函数（可学习窗函数）
- 音频特征提取（神经特征）
- 动态范围处理（AI自适应压缩）

### 3. AI音频处理 (ai-audio.html) - 新增
- Transformer架构
- 扩散模型（AudioLDM、MusicGen）
- 神经音频编解码器
- 实时AI语音增强（RNNoise、DeepFilterNet）
- 预训练模型（Whisper、Wav2Vec 2.0）

### 4. Python示例 (python-examples.html)
- Whisper语音识别
- RNNoise实时降噪
- MusicGen音乐生成
- Wav2Vec特征提取
- Demucs音乐分离
- ONNX模型部署

### 5. Golang示例 (golang-examples.html)
- AI优化WAV处理
- ONNX模型推理引擎
- 实时音频处理管道
- AI音频特效处理

## 年新工具与框架

### Python
```bash
# AI/深度学习
pip install torch torchaudio torchvision
pip install transformers datasets
pip install speechbrain
pip install audiolm-pytorch audiocraft
pip install onnxruntime onnx

# 实时音频
pip install sounddevice noisereduce
```

### Golang
```bash
go get github.com/yourbasic/onnxruntime
go get github.com/second-state/go-onnxruntime
go get github.com/owulveryck/onnx-go
```

## 年技术趋势

- **端侧AI音频**：轻量级模型在设备上运行
- **多模态音频**：音频+文本+视觉联合处理
- **实时生成**：低延迟AI音频生成
- **神经编解码器**：10x压缩率
- **个性化音频**：用户自适应处理

## 使用方法

1. **查看网站**: 直接在浏览器中打开 `index.html` 文件
2. **导航**: 使用顶部导航栏在各个章节之间切换
3. **代码示例**: 点击对应的代码块查看语法高亮的示例代码

## 技术特点

- 📱 **响应式设计**: 适配各种设备尺寸
- 🎨 **现代化UI**: 使用CSS渐变和动画效果
- 📊 **可视化**: 使用SVG和图表展示概念
- 💻 **代码高亮**: 专业的代码着色显示
- 🤖 **AI集成**: 包含最新的AI音频处理技术

## 学习建议

1. **按顺序学习**: 建议从基础开始，逐步深入到AI处理
2. **理论结合实践**: 每学完一个概念就动手实现
3. **运行代码**: 尝试运行示例代码加深理解
4. **关注前沿**: 关注arXiv最新论文和技术博客

## 许可证

本项目仅供学习使用。

## 参考资源

- [Hugging Face Audio](https://huggingface.co/audio)
- [PyTorch Audio](https://pytorch.org/audio/)
- [SpeechBrain](https://speechbrain.github.io/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [MusicGen](https://github.com/facebookresearch/audiocraft)
- [Demucs](https://github.com/facebookresearch/demucs)
