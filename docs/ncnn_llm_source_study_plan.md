# ncnn_llm 源码研读计划

> **目标**：先跑通一个小模型，再从源码层面还原完整的 LLM 推理主链路：  
> `Model Loading → Prefill → Decode → KV Cache → Token Sampling`
>
> **学习方式**：以可运行实验为起点，以源码调用链为主线；每个阶段都包含学习目标、源码阅读任务、实践任务和验收标准。
>
> **预计投入**：约 12～18 小时。该时间仅供规划，不以赶进度为目标；遇到模型转换、算子或缓存机制等难点时，可以延长。

---

## 一、总体路线

| 阶段 | 学习主题 | 核心问题 | 主要交付物 |
|---|---|---|---|
| Phase 0 | 环境与仓库准备 | 如何成功构建并运行项目？ | 环境记录、固定版本、可运行命令 |
| Phase 1 | Model Loading | 模型如何从文件变成可执行的网络？ | 模型加载调用图 |
| Phase 2 | Prefill | 输入文本如何转化为模型的初始计算状态？ | Prefill 时序图 |
| Phase 3 | Decode | 模型如何逐个生成 Token？ | Decode 调用图 |
| Phase 4 | KV Cache | 如何复用历史计算结果？ | KV Cache 数据流图与实验记录 |
| Phase 5 | Token Sampling | 模型如何从输出分数中选择下一个 Token？ | 采样流程图与对比实验 |
| Final | 全链路整合 | 如何将所有阶段对应到真实源码？ | 完整调用图、源码索引、学习总结 |

### 总体主链路

```text
Model Loading
     │
     ▼
Tokenizer / 输入准备
     │
     ▼
Prefill
     │
     ▼
Logits
     │
     ▼
Token Sampling
     │
     ▼
新 Token
     │
     ▼
Decode ───────────────┐
     │                │
     ▼                │
读取 / 更新 KV Cache │
     │                │
     └── 继续生成 ◄───┘
          │
          ▼
      停止条件满足
          │
          ▼
        输出文本
```

> 这是概念图，不是已经核实的源码时序图。实际调用关系、函数名和执行顺序必须以所选仓库版本为准。模型加载通常发生在生成循环之前；Tokenizer 的初始化位置也需要通过源码确认。

---

## 二、Phase 0：环境与仓库准备

**预计投入：1～2 小时**

### Task 0.1：确认仓库与版本

优先检查以下仓库：

- 原始项目：<https://github.com/futz12/ncnn_llm>
- 候选上游仓库：<https://github.com/nihui/ncnn_llm>
- ncnn 官方仓库：<https://github.com/Tencent/ncnn>

先确认原始仓库是否仍可访问，以及它与候选上游仓库的关系。不要在未经核实的情况下假定两者代码相同。

可以执行：

```bash
git ls-remote https://github.com/futz12/ncnn_llm.git HEAD
git ls-remote https://github.com/nihui/ncnn_llm.git HEAD
```

确定使用哪个仓库后，克隆并记录 commit：

```bash
git clone <确认后的仓库地址>
cd <仓库目录>
git rev-parse HEAD
```

具体构建方式、依赖和命令以该版本的 README 为准。

**记录内容：**

- 仓库地址和 commit。
- 操作系统、CPU、内存。
- 编译器和构建工具版本。
- 是否启用 Vulkan，以及设备和驱动支持情况。
- 模型文件来源、格式和许可信息。

### Task 0.2：成功运行一个小模型

选择当前版本明确支持、容易获取且适合本机的小模型。如果所选版本支持 Qwen3 0.6B，可将其作为候选；实际模型格式、下载方法和命令必须以仓库文档为准。

目标：

1. 准备模型文件。
2. 编译项目。
3. 执行一次文本生成。
4. 保存运行命令和终端输出。
5. 记录大致启动时间、生成速度和内存占用。

示例命令（仅在对应版本确实支持这些参数时使用）：

```bash
xmake run llm_ncnn_run \
  --model ./assets/qwen3_0.6b
```

**验收标准：**

- 程序成功启动。
- 模型成功加载。
- 能够输入文本并生成非空输出。
- 同一命令可以重复执行。
- 运行命令、模型版本和 commit 均有记录。

**阶段交付物：**

```text
study/
├── environment.md
├── repository.md
└── source-map.md
```

---

## 三、Phase 1：Model Loading——模型如何加载

**预计投入：2～3 小时**

### Task 1.1：从程序入口追踪调用链

找到 CLI 程序入口，逐步追踪：

1. `main()` 或实际入口在哪里？
2. 命令行参数在哪里解析？
3. `--model` 参数传给了哪个对象？
4. 哪个函数创建或初始化模型运行时？
5. 哪个函数真正加载网络与权重？
6. Tokenizer 在哪里初始化？

建立**经源码验证**的调用图，例如：

```text
程序入口
   │
   ▼
参数解析
   │
   ▼
运行时初始化
   │
   ▼
读取模型配置
   │
   ▼
加载网络与权重
   │
   ▼
Tokenizer 初始化（具体位置待验证）
```

不要只凭函数名推断职责。通过调用点、函数实现和调试器确认实际流程。

### Task 1.2：理解模型配置、权重与运行时

重点理解：

| 概念 | 作用 |
|---|---|
| 模型配置 | 描述模型结构、参数或其他运行信息 |
| 模型权重 | 保存训练得到的参数 |
| 推理运行时 | 组织输入、执行网络计算并生成输出 |
| Tokenizer | 在文本与 Token ID 之间转换 |

需要检查：

- 模型配置文件的实际格式。
- 网络结构和权重文件如何加载。
- 模型架构差异如何处理。
- 发生加载失败时，错误如何返回或记录。

**实践任务：**

在关键函数中添加日志或设置断点，记录函数进入顺序、模型目录、配置文件、权重文件以及 Tokenizer 初始化位置。

**验收标准：**

能够从入口追踪到模型加载完成，并用源文件、函数名和调用顺序支持自己的结论。

**阶段交付物：** `model-loading.md`，包含调用图、关键函数索引和加载过程说明。

---

## 四、Phase 2：Prefill——模型如何处理输入

**预计投入：2～3 小时**

### Task 2.1：从文本追踪到 Token IDs

理解以下流程：

```text
原始文本
   │
   ▼
Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Embedding
   │
   ▼
Transformer
```

研究：

- Tokenizer 如何读取词表和配置。
- 文本如何转换为 Token ID。
- 是否添加 BOS、EOS 等特殊 Token。
- Token ID 如何映射为 Embedding。
- 输入序列长度如何计算。

**实践任务：**

对以下输入观察 Tokenizer 输出，并记录 Token ID 与序列长度：

```text
Hello
Hello world
Hello world!
```

不要求在这一阶段深入所有 Tokenizer 算法，重点是弄清楚当前项目的实际处理方式。

### Task 2.2：追踪 Prefill 计算

Prefill 通常会处理整个输入序列，建立后续生成所需的计算状态。

概念流程：

```text
Token IDs
    │
    ▼
Token Embedding
    │
    ▼
Transformer Layers
    │
    ▼
KV Cache 初始化 / 更新
    │
    ▼
Logits
```

需要从源码确认：

1. 输入 Token 在哪里进入模型。
2. Transformer 层如何被调用。
3. Attention 在哪里执行。
4. Prefill 如何初始化或更新 KV Cache。
5. 哪个位置生成 Logits。
6. 实际实现计算全部位置的输出，还是只取最后一个位置用于下一 Token 预测。

**实践任务：**

记录关键张量的形状、输入序列长度、缓存状态变化，以及 Prefill 结束时得到的输出。

**验收标准：**

能够解释 Prefill 的输入、主要计算步骤、缓存变化，以及它如何为生成第一个新 Token 做准备。

**阶段交付物：** `prefill.md`，包含时序图、关键函数和张量形状记录。

---

## 五、Phase 3：Decode——模型如何逐个生成 Token

**预计投入：2～3 小时**

### Task 3.1：追踪一次 Decode

在常见的增量 Decode 中，模型通常只处理最新 Token，并读取历史 KV Cache；具体行为必须通过所选版本的源码确认。

概念流程：

```text
当前 Token
    │
    ▼
Embedding
    │
    ▼
Transformer Layers + 历史 KV Cache
    │
    ▼
Logits
    │
    ▼
选择下一个 Token
```

设置断点，观察连续生成的几个 Token。每一步记录：

- 当前输入 Token。
- 当前序列位置。
- 输入张量形状。
- Logits 形状。
- 返回的 Token ID。
- KV Cache 是否被读取或更新。

重点问题：Prefill 与 Decode 是否调用同一套 Transformer 计算函数？它们如何区分整段输入与增量输入？

### Task 3.2：理解生成循环

以下为概念伪代码，不代表仓库的实际实现：

```cpp
prefill(prompt_tokens);

while (!finished) {
    logits = decode(current_token);
    next_token = select_token(logits);

    if (is_eos(next_token)) {
        break;
    }

    output(next_token);
    current_token = next_token;
}
```

在源码中进一步确认：

- 生成循环在哪里。
- `decode()` 的真实调用关系。
- 何时更新当前 Token 和位置。
- EOS、停止词和最大长度如何处理。
- 采样逻辑是否独立于 Decode。
- 异常或无效输出如何处理。

**验收标准：**

能够解释 Prefill 与 Decode 的职责差异，并通过断点验证连续 Token 的生成过程。

**阶段交付物：** `decode.md`，包含单步 Decode 流程、生成循环和关键状态变化。

---

## 六、Phase 4：KV Cache——如何复用历史计算结果

**预计投入：3～4 小时**

这是整条学习主线中值得重点投入的阶段。

### Task 4.1：理解 KV Cache 的基本原理

Attention 会使用历史 Token 对应的 Key 和 Value。增量生成时，运行时可以保存先前计算的 K/V，供后续 Token 复用。

概念示意：

```text
处理 Token 1
    │
    ▼
计算 K/V 并缓存
    │
    ▼
处理 Token 2
    │
    ├── 计算当前 K/V
    ├── 读取历史 K/V
    └── 更新缓存
             │
             ▼
         处理 Token 3
```

实际实现通常需要按层维护缓存，也可能使用不同的内存布局。

需要回答：

1. Key 和 Value 在 Attention 中承担什么作用？
2. 为什么历史 K/V 可以复用？
3. 为什么通常需要为多个 Transformer 层维护缓存？
4. 新 Token 的 K/V 在何时写入？
5. Attention 如何读取历史缓存？
6. 序列长度增长时，缓存内存如何变化？

### Task 4.2：定位真实缓存实现

在源码中寻找：

- Cache 数据结构的声明。
- 缓存的初始化与分配。
- Prefill 时的缓存写入。
- Decode 时的缓存读取。
- 新 Token 对应缓存的更新。
- 位置、长度和容量的维护。
- 超出上下文长度时的处理方式。

画出实际的数据流图，并为每个节点标注源文件和函数。

### Task 4.3：通过实验验证 KV Cache

在项目允许的前提下，对比：

- 正常使用 KV Cache 的 Decode。
- 受控条件下重新计算历史上下文的参考实现。

观察：

- 输出是否一致。
- 单步生成时间如何变化。
- 输入长度增加时耗时如何变化。
- 缓存内存占用如何变化。

公平比较时，尽量保持模型、输入、计算精度、输出选择规则和硬件环境一致。浮点计算差异也可能影响最终生成结果。

如果项目没有现成的无缓存模式，不要为了实验直接大改核心代码。可以先通过源码分析和日志验证缓存读写，再决定是否实现独立参考路径。

**验收标准：**

能够指出缓存的数据结构及读写位置，解释缓存为何能减少重复计算，并说明它的内存成本和适用边界。

**阶段交付物：** `kv-cache.md`，包含原理说明、源码位置、数据流图与实验记录。

---

## 七、Phase 5：Token Sampling——如何选择下一个 Token

**预计投入：2～3 小时**

### Task 5.1：理解 Logits 与概率

模型通常会输出一个 Logits 向量。Logits 是未归一化分数，不是概率；Softmax 可以将其转换为概率分布。

例如：

| Token | Logits |
|---|---:|
| A | 2.1 |
| B | 0.5 |
| C | 3.8 |
| D | -0.4 |
| E | 1.2 |

使用 Greedy Decoding 时，选择 Logits 最大的 Token，即 Token C。

### Task 5.2：研究不同选择策略

| 策略 | 基本思想 |
|---|---|
| Greedy | 每次选择分数最高的 Token |
| Temperature | 调整概率分布的尖锐程度 |
| Top-K | 只考虑分数最高的 K 个候选 |
| Top-P | 选择累计概率达到指定阈值的候选集合 |

在源码中检查：

1. Logits 在哪里产生。
2. 默认使用哪种选择策略。
3. Temperature 在哪里应用。
4. 是否实现 Top-K、Top-P。
5. 使用什么随机数生成器。
6. EOS 和停止条件在什么位置检查。
7. 采样与输出 Token 的代码是否分离。

**实践任务：**

在相同模型和输入下，对比 Greedy 与采样模式的输出。记录参数、随机种子（如果支持）以及输出差异。

注意，Temperature、Top-K 和 Top-P 的处理顺序可能影响结果，应以实际代码为准。

**验收标准：**

能够从 Logits 追踪到最终 Token ID，并解释不同选择策略如何影响输出。

**阶段交付物：** `token-sampling.md`，包含采样流程图、源码索引和对比实验。

---

## 八、Final：整合完整推理调用链

将各阶段的发现串起来，形成**源码对应版本**的完整流程图。

```text
用户输入
   │
   ▼
Tokenizer / 输入准备
   │
   ▼
Token IDs
   │
   ▼
Prefill
   │
   ├── Transformer 计算
   ├── 初始化 / 更新 KV Cache
   └── 得到用于下一 Token 预测的 Logits
                 │
                 ▼
            Token Sampling
                 │
                 ▼
              新 Token
                 │
                 ▼
               Decode
                 │
          ┌──────┴──────┐
          ▼             ▼
      更新 KV Cache   检查停止条件
          │             │
          └── 继续生成  ├── 未结束：继续
                        └── 已结束：输出文本
```

> 这张图是最终整理的目标，不是仓库已核实的实现图。请根据源码修正调用顺序、缓存操作位置以及停止条件。

### 最终验收清单

- [ ] 从程序入口追踪到模型加载函数。
- [ ] 解释模型配置、权重与运行时的关系。
- [ ] 说明文本如何变成 Token IDs。
- [ ] 说明 Prefill 和 Decode 的区别。
- [ ] 找到 KV Cache 的数据结构、读取与更新位置。
- [ ] 解释 Logits 如何转换为下一个 Token。
- [ ] 在源码中定位完整生成循环。
- [ ] 使用调试器验证关键函数的调用顺序。
- [ ] 记录至少一项与缓存或生成行为相关的实验。
- [ ] 形成一张标注真实源文件与函数名的完整调用图。

---

## 九、建议的源码阅读与调试方法

### 阅读顺序

1. CLI 入口与参数解析。
2. 运行时初始化和模型加载。
3. Tokenizer 与输入预处理。
4. Prefill 入口。
5. Transformer / Attention 计算。
6. KV Cache 的数据结构与读写。
7. Decode 与生成循环。
8. Logits 处理和 Token Sampling。
9. 停止条件及输出处理。

不要一开始就通读所有模型实现。先沿一条真实可运行的路径追踪，再在必要时深入模型特有的分支。

### 每个关键函数都记录四件事

| 项目 | 记录内容 |
|---|---|
| 输入 | 参数、张量形状、状态 |
| 处理 | 函数的核心职责 |
| 输出 | 返回值、状态变化 |
| 调用关系 | 谁调用它、它又调用谁 |

### 建议使用的工具

- IDE 调试器：查看调用栈、变量和张量形状。
- 日志：记录关键函数的进入顺序和状态变化。
- Git：固定版本、追踪修改。
- 简单基准脚本：记录耗时和内存变化。

先理解执行流程，再考虑引入复杂性能分析工具。

---

## 十、参考仓库与资料

- 原始项目：<https://github.com/futz12/ncnn_llm>
- 候选上游仓库：<https://github.com/nihui/ncnn_llm>
- ncnn 官方仓库：<https://github.com/Tencent/ncnn>
- ncnn 官方 Wiki：<https://github.com/Tencent/ncnn/wiki>

在开始源码分析前，先确认目标仓库的可访问性、版本关系、commit、模型支持和构建说明。模型许可也需要单独核实，不能仅凭代码仓库的许可证判断模型权重是否可自由使用。

---

## 十一、下一步

**现在只执行 Phase 0，不必同时开展所有阶段。**

1. 检查原始仓库和候选上游仓库是否可访问。
2. 确定研究仓库并固定 commit。
3. 按该版本的说明完成构建。
4. 运行一个受支持的小模型。
5. 保存运行命令、终端输出和环境信息。

完成后，再从程序入口开始追踪调用链，一次确认一个节点，逐步还原：

`Model Loading → Prefill → Decode → KV Cache → Token Sampling`

**原则：每个结论都尽量由真实源码、调试记录或可复现实验支持；示意图和伪代码不能替代对实际实现的验证。**
