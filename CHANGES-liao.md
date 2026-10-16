# 本 fork 相对上游 PyFR 的改动记录（liao / idea 分支）

上游基准：PyFR/PyFR `develop`，分叉点 `a20e39d9`（截至 2026-10-16 上游无新提交）。
分支关系：`liao`（主干）→ `idea/wmles-es`（熵稳定格式 + 壁模型）→ `idea/wm-volsrc`（仅测试，未合并）。
所有新功能都需在配置中显式开启；不开启时，Euler/NS 与 `liao` 逐位相同（同线程数），壁模型默认路径与改动前逐位相同。

状态标记：
- **有效**：算例证明需要或有明显改善，正式使用
- **工具**：诊断/测试用，完成了它的用途，不是方法本身
- **无效**：算例证明没有用或有害
- **未用**：实现正确，但没在正式算例中使用或尚未证明有收益

算例与证据位于 `examples_src/`（不属于本仓库）；下文给出算例名，详见 `examples_src/wmles/dev/overnight_report.md` 与 `report/wmles_report.pdf`。

---

## 一、`liao` 相对上游（4 个文件）

| 改动 | 文件 / 提交 | 说明 | 状态 |
|---|---|---|---|
| SLAU 黎曼解 | `euler/kernels/rsolvers/slau.mako`，文档 `solver-interfaces.rst`；`f79e8a2f` | 全速域 AUSM 族通量 | **有效**：4 层网格 WMLES 中对数区 uu⁺、ww⁺ 偏高约减半（`s_slau_*`）。注意：非熵稳定 |
| tvd-rk3 第三步权重在工作精度下和为 1 | `integrators/explicit/steppers.py`；`9adaf579` | 单精度下 fl(2/3)+fl(1/3)≠1，每步放大守恒量 | **有效**（FP32 生产计算；FP64 下无影响） |
| 湍流入口插件补回 clip 系数 | `plugins/solver/kernels/turbulence.mako`；`de07b7b4` | 上游丢失 clip，注入强度与设定无关 | **有效**（M=2 TBL：ti 1/4 能量比由 0.99 恢复为 15.8，期望 16） |

## 二、`idea/wmles-es` 相对 `liao`（20 个文件）

### 2.1 熵稳定体积通量（`[solver] volume-flux = entropy-conservative`）

| 改动 | 文件 / 提交 | 说明 | 状态 |
|---|---|---|---|
| 熵守恒通量差分（Chan 2018 解耦 SBP，Gauss 点） | `euler/elements.py`（`_set_ec_kernels`）、`kernels/ecflux.mako`、`entvar.mako`、`entproj.mako`、`tfluxdiff.mako`；`a2435d2c` `151211eb` | 仅六面体、Gauss 点、仿射单元、FP64 | **有效**：同设置下标准格式 t≈11 发散，ES 全部算例稳定到 t=400，熵单调减 |
| NS 中无粘部分走通量差分 | `navstokes/elements.py`、`navstokes/kernels/tflux.mako`；`48e045f9` | tflux 只算粘性部分 | **有效** |
| 散度经外部项加进右端 | `baseadvec/elements.py`、`baseadvec/kernels/negdivconf.mako`（`esdiv`） | | **有效** |
| 计算图依赖：tdivtconf/tdivtpcorf 排在 tfluxdiff 之后 | `baseadvec/system.py`、`baseadvecdiff/system.py`；`47281505` `48e045f9` | 修复 uin==fout 时的读写竞争（CUDA 上 RK4 出 NaN） | **有效**（必要的修复） |
| `ec-rusanov` 界面通量 | `rsolvers/ec-rusanov.mako`；`a375ba16` | 熵守恒 + LF 型耗散 | **有效**（所有 ES 正式算例） |
| `ec` 纯熵守恒界面通量 | `rsolvers/ec.mako` | | **工具**（熵守恒检验、分项诊断） |
| ~~按解点并行的 GPU 内核 `tfluxdiffp`~~ | 已删除（原 `e0c9f5d9`） | CUDA 上实测无加速 | **已删除**；GPU 上现与 CPU 一样用按单元并行的 `tfluxdiff` |

### 2.2 壁模型边界 `wm-adia-wall`（`navstokes/inters.py`、`kernels/bcs/wm-adia-wall.mako`；`749c6a7d` 起）

| 选项 | 说明 | 状态 |
|---|---|---|
| 基本框架：镜像滑移 ghost + τ_w 动量通量，采样表与视图 | | **有效** |
| `sample = point`，`sample-point = k` | 取贴壁单元第 k 个解点 | **有效但被取代**：τ_w 随 k 在 −20%～+7.5%（4 层）间变化（`s_es400_k1..k4`）；保留作对照 |
| `sample = column-mean`（`5b748fb3`） | 整列加权平均 + 壁律解析积分平均 | **有效**：τ_w +2.2%（4 层）、+0.6%（8 层）、+0.6%（p=4），不依赖采样点（`s_es400_cm`、`s8_es400_cm`、`s_p4_cm`） |
| `wall-law = log-exp` | LOG-EXP 壁律（Zhang 等 arXiv:2104.09054 式 10） | **有效** |
| `wall-law = linear` | 线性律 | **工具**（内核检验） |
| `wall-law = fixed`，`tau-fixed`（`cf40307c`） | τ_w 大小固定 | **工具**：证明对数区应力偏差与 τ_w 大小基本无关（`s_es400_fix`） |
| `wall-law = zero-stress`（原名 `none`，`5bb4de1b` 改名） | τ=0，零切应力滑移壁，不是无滑移 | **工具**（分项诊断中分离 τ_w 贡献） |
| ~~`inviscid-state = noslip`~~ | 已删除 | 曾证明无效：Rusanov 下壁面力符号错误，HLLC 下发散（3.1 标准格式） |
| ~~`damp-k`、`slip-ratio`~~ | 已删除 | 曾证明无效：该项与 τ_WM 抵消，壁面合力≈0（3.1 标准格式） |
| `debug` | 输出采样状态与 τ | **工具**（内核检验） |

### 2.3 恒流量驱动插件 `flowrate`

| 文件 | 说明 | 状态 |
|---|---|---|
| `plugins/solver/flowrate.py`、`kernels/flowrate.mako`、`plugins/solver/__init__.py` | 体积流量恒定，驱动力自动调节 | **有效**（所有槽道算例） |

### 2.4 上游 bug 修复

| 文件 / 提交 | 说明 | 状态 |
|---|---|---|
| `backends/base/mathfns.py`；`d895cdb6` | OpenMP 后端 `_pyfr_exp` 在 |x|>≈708 时出错（exp(−723) 得 −3.3e302）；先截断再分两步缩放 | **有效**（修复前壁模型 k=3 的 τ 偏差 15%） |

## 三、`idea/wm-volsrc` 相对 `idea/wmles-es`（仅测试，未合并；4 个文件）

| 改动 | 说明 | 状态 |
|---|---|---|
| `stress-deposit = face` | 默认：τ_w 作面通量，经提升按 −l_a(−1) 交替注入各解点 | 对照 |
| ~~`stress-deposit = volume`~~（原 `7abd9574`） | 已删除 | 曾证明有害：近壁解析应力被压没、湍流强度降约 38%、τ_w +8.7%（`s_es400_cmvol`） |
| `stress-deposit = first-point`（`3e1d2bc5`） | τ_w 全部作第一离壁解点上的汇 | **有效**：第一层速度误差 p=3 −11.8%→−3.0%，p=4 −5.9%→+0.9%；对数区 −uv⁺、uu⁺ 偏高约减半（`s_es400_cmfp`、`s_p4_cmfp`）。尚无稳定性证明 |
| `negdivconf` 外部源 `wmsrc`、`baseadvec/elements.py` 开关 | 为第一离壁点服务，默认关闭 | 随第一离壁点 |
| `wall-law = none` 置零修复（`2bd12b0b`） | 7abd9574 误删了 τ=0 | 修复 |

## 四、已被算例证明无效并已删除的改动（2026-10-16）

1. `tfluxdiffp`（GPU 按解点并行内核）：无加速。
2. `inviscid-state = noslip`：壁面力错误或发散。
3. `damp-k` / `slip-ratio`：与 τ_WM 抵消，壁面合力≈0。
4. `stress-deposit = volume`（均匀分摊，测试分支）：压制近壁湍流，τ_w 变差。

删除后的回归（均与删除前逐位或逐字一致）：壁模型单点默认路径内核检验（k=1–4）、列平均内核检验、
4 层网格整算例 t=10 解场（五个守恒量逐位相同）、熵守恒内核检验 level3（同线程数对比）。

另有 `sample = point` 虽能工作，但已被 `column-mean` 取代；`wall-law = fixed/zero-stress/linear`、`ec`、`debug` 为诊断工具。
