# `main.py` 讲解

这份脚本的核心流程是：

1. 读取本组配置 `group_config.json`
2. 读取本组输入数据 `input_cases.json`
3. 生成一批用于训练的合成车载流量样本
4. 用随机森林做训练
5. 对本组输入窗口做预测
6. 计算风险分数、异常包数量等展示指标
7. 输出 `predictions.json`、`run_summary.txt`
8. 画三张图

## 1. 开头导入部分

- `from __future__ import annotations`
  作用是让类型注解延迟解析，方便写 `list[dict[str, object]]` 这种类型提示。
- `json`、`csv`、`Path`
  用来读写 JSON、CSV 和处理路径。
- `numpy`
  做数值计算和矩阵处理。
- `RandomForestClassifier`
  这是随机森林模型本体。
- `matplotlib`
  用来画图。

## 2. `FEATURES` 和 `LABELS`

- `FEATURES`
  这是模型输入特征，比如平均流量、帧率、重放比例、模糊比例、欺骗比例、UDS 比例。
- `LABELS`
  这是模型输出类别，也就是最终预测结果：`安全`、`DoS`、`重放`、`模糊`、`欺骗`、`UDS非法会话`。

## 3. `CLASS_PROFILES`

这里定义每一类攻击的大致"画像"：

- `DoS`：高流量、高帧率、高突发
- `重放`：`replay_ratio` 更高
- `模糊`：`fuzzy_ratio` 和 `payload_entropy` 更高
- `欺骗`：`spoof_ratio` 更高
- `UDS非法会话`：`uds_ratio` 更高

每个值都是 `(均值, 标准差)`，脚本会围绕这些统计特征去造训练样本。

## 4. `FEATURE_ADJUSTMENTS`、`CONFUSION_MAP`、`DATA_PROFILE_DEFAULTS`

- `FEATURE_ADJUSTMENTS`
  让不同类别更像自己，比如 DoS 再提高流量和帧率。
- `CONFUSION_MAP`
  定义哪些类别容易混淆，比如 `重放` 和 `UDS非法会话`。
- `DATA_PROFILE_DEFAULTS`
  控制数据多样性，比如噪声强度、边界样本比例、漂移样本比例、标签噪声比例。

这些东西的作用，是避免模型学到"过于干净"的数据。

## 5. 训练数据是怎么生成的

- `sample_feature_value()`
  生成单个特征值。
- `apply_label_signature()`
  把一条样本往某种攻击的特征方向再推一把。
- `generate_training_rows()`
  批量生成训练集。

这里最关键的是：脚本会故意加入边界样本、漂移样本和少量标签噪声，所以准确率不会总是 1。

## 6. 随机森林真正起作用的地方

在 `train_model()` 里：

- `train_test_split(...)`
  把数据拆成训练集和测试集。
- `RandomForestClassifier(**group_config["rf_params"])`
  创建随机森林模型。
- `model.fit(x_train, y_train)`
  训练模型。
- `model.predict(x_test)`
  在测试集上做预测。
- `accuracy_score(...)`
  算准确率。

### 随机森林参数怎么理解

- `n_estimators`
  森林里有多少棵树。
- `max_depth`
  每棵树最多长多深。
- `min_samples_split`
  一个节点最少多少样本才继续分裂。
- `min_samples_leaf`
  叶子节点最少保留多少样本。
- `max_features`
  每次分裂时随机看多少个特征。
- `class_weight`
  类别不均衡时怎么加权。
- `random_state`
  固定随机种子，保证结果可复现。

## 7. 预测部分在做什么

- `predict_cases()`
  把本组 `input_cases.json` 喂给模型，得到预测标签和类别概率。
- `compute_case_metrics()`
  这一步不是随机森林本体，而是把预测结果进一步转换成风险分数、异常包数量、低中高等级拆分，用来做可视化。

## 8. 三张图怎么来的

- `render_status_chart()`
  画正常流量和异常攻击的环形图。
- `render_time_series_chart()`
  画时间分布折线图。
- `render_stacked_bar_chart()`
  画攻击类型 / 等级堆叠柱图。

## 9. 摘要输出

- `make_run_summary()`
  汇总参数、样本规模、类别分布、准确率、置信度区间、风险分数区间、误判数。
- `format_run_summary()`
  把这些信息格式化成终端可读文本。

## 10. `main()` 做了什么

`main()` 就是总调度器：

1. 读配置
2. 读输入
3. 生成训练数据
4. 训练随机森林
5. 预测
6. 保存 CSV / JSON / 图片
7. 打印摘要

## 11. 你可以怎么讲

你可以用一句话概括：

> 这段代码先构造带噪声和边界效应的车载 IDS 训练数据，再用随机森林学会区分安全、DoS、重放、模糊、欺骗和 UDS 非法会话，最后把本组输入窗口的预测结果转换成图表，方便展示。
