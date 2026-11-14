# 文档索引（docs/）

这里是**项目文档的唯一入口**。根目录只放 `README.md`（读者第一眼要看的东西）
与 `VERSION` / `LICENSE`；专题与历史材料统一收进 `docs/`，就近的实现契约则
留在各自代码目录旁边（见下表）。

## 放哪儿

| 类型 | 位置 |
| --- | --- |
| 架构、格式、API、启动、目录树总览 | `../README.md` |
| 文档索引与归档规则 | `docs/README.md`（本文件） |
| 历史评估 / 已作废的方案稿 | `docs/`，文件名保留原名，正文顶部必须写明「历史记录，请勿据此实施」 |
| 与某个模块绑定的实现契约 | 该模块目录内，例如 `../service/web/plugins/*.md`、`../service/prims/README.md` |
| 目录级约定 | 该目录内，例如 `../environment/README.md`、`../service/README.md` |
| 后续测试 / 对比数据 | `../experiments/`（只放数据，`README.md` 说明约定） |

## 现有文档

| 文档 | 职责 | 权威性 |
| --- | --- | --- |
| [`../README.md`](../README.md) | 架构、token 系统、训练/推理、启动、HTTP 契约、文件树 | 现行 |
| [`VOCAB_EXPANSION_EVAL.md`](./VOCAB_EXPANSION_EVAL.md) | base1.2.0 时代的词表扩展评估稿 v2 | **历史**：结论未实施，1.3.0 走的是另一套口径 |
| [`../service/web/plugins/README.md`](../service/web/plugins/README.md) | 推理显示插件协议（8 节） | 现行 |
| [`../service/web/plugins/AGENT_TOKENS.md`](../service/web/plugins/AGENT_TOKENS.md) | 契约头 / core / slot / typst / 段类型全表 | 现行 |
| [`../service/web/plugins/TYPST_TOKENS.md`](../service/web/plugins/TYPST_TOKENS.md) | 114 构造 × 开闭对 = 228 token 对应表 | 由 `gen_typst_map.py` 生成，禁止手改 ID |
| [`../service/prims/README.md`](../service/prims/README.md) | 图元管线 GPU / CPU 两条路径与环境变量 | 现行 |
| [`../service/README.md`](../service/README.md) | 服务目录约定与 `output/` 产物规则 | 现行 |
| [`../environment/README.md`](../environment/README.md) | 零系统依赖契约、一个依赖一个文件夹 | 现行 |
| [`../start/requirements/README.md`](../start/requirements/README.md) | 两份依赖清单与唯一安装入口 | 现行 |
| [`../experiments/README.md`](../experiments/README.md) | 预留目录的用途与命名约定 | 现行 |

## 双向对齐约定

1. 架构常量只在 `server/spec.py`，路径只在 `server/paths.py`；文档只描述、
   不从文档里抄回代码。前端一律从 `/api/agent/spec`、`/api/status`、
   `/api/train/dataset/spec` 读 ID 与布局。
2. 改了契约头 / 分区，必须同步 `vfc.head_entries()`、`build_vocab.build_entries()`、
   词表 `.vfc` + `config.json`、关于页 SVG、`minify.py` 重建与相关测试。
3. 改了 `service/web/index_source.html` 必须重跑 `service/web/minify.py`
   再 `--check`；只改 `app.css` / `app.js` 不用重建。
4. 每次架构性调整都要同步「关于」页；版本徽章读根 `VERSION`。
5. 历史稿只增不改结论；若最终实施，应在根目录 `README.md` 写现行口径，
   并在历史稿顶部加「已由 X 取代」的链接。
