# 本 fork 相对上游 PyFR 的改动记录（liao / idea 分支）

上游基准：PyFR/PyFR `develop`，分叉点 `a20e39d9`（截至 2026-10-16 上游无新提交）。
分支关系：`liao`（主干）→ `idea/wmles-es`（熵稳定格式 + 壁模型；2026-10-16 清理后合并进 `liao`）→ `idea/wm-volsrc`（仅测试，未合并）。
合并前检验：标准 NS（不开新选项、上游 slp-adia-wall）同一随机状态下面通量与 RHS 与合并前的 `liao` 逐位相同。
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
| 熵守恒通量差分（Chan 2018 解耦 SBP，Gauss 点） | `euler/elements.py`（`_set_ec_kernels`，一维算子取自 PyFR 线基）、`kernels/ecflux.mako`、`entvar.mako`、`entproj.mako`、`tfluxdiff.mako`；`a2435d2c` `151211eb` | 仅六面体、Gauss 点、仿射单元、FP64 | **有效**：同设置下标准格式 t≈11 发散，ES 全部算例稳定到 t=400，熵单调减 |
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
| LOG-EXP 壁律 | Zhang 等 arXiv:2104.09054 式 10；唯一壁律，不再有 `wall-law` 选项 | **有效** |
| ~~`wall-law = linear`~~ | 已删除 | 工具，用途已完成（内核检验） |
| ~~`wall-law = fixed`，`tau-fixed`~~（原 `cf40307c`） | 已删除 | 工具，用途已完成：证明对数区应力偏差与 τ_w 大小基本无关（`s_es400_fix`） |
| ~~`wall-law = zero-stress`~~（原名 `none`） | 已删除 | 与上游 `slp-adia-wall` 的 RHS 逐位相同（rusanov/ec-rusanov、point/column-mean 均验证），诊断改用上游边界 |
| ~~`inviscid-state = noslip`~~ | 已删除 | 曾证明无效：Rusanov 下壁面力符号错误，HLLC 下发散（3.1 标准格式） |
| ~~`damp-k`、`slip-ratio`~~ | 已删除 | 曾证明无效：该项与 τ_WM 抵消，壁面合力≈0（3.1 标准格式） |
| ~~`debug`~~ | 已删除 | 内核检验改为读格式实际使用的面通量。打开它会改变编译器的 FMA 合并，使 τ 差 1 ulp，测量通道本身干扰被测量 |

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

## 四、已被算例证明无效并已删除的改动（2026-10-16）

1. `tfluxdiffp`（GPU 按解点并行内核）：无加速。
2. `inviscid-state = noslip`：壁面力错误或发散。
3. `damp-k` / `slip-ratio`：与 τ_WM 抵消，壁面合力≈0。
4. `stress-deposit = volume`（均匀分摊，测试分支）：压制近壁湍流，τ_w 变差。

删除后的回归（均与删除前逐位或逐字一致）：壁模型单点默认路径内核检验（k=1–4）、列平均内核检验、
4 层网格整算例 t=10 解场（五个守恒量逐位相同）、熵守恒内核检验 level3（同线程数对比）。

另有 `sample = point` 虽能工作，但已被 `column-mean` 取代；`ec` 为诊断工具。

## 五、整洁性清理（2026-10-16）

1. 删除诊断桩：`wall-law`（`linear`/`fixed`/`zero-stress`）、`tau-fixed`、`debug`；壁模型类不再保存只给测试用的属性（`_hwm`、`_uwms`、`_uwnames`、`_wmcols`）。
2. `_set_ec_kernels` 的一维求导矩阵与端点插值改用 PyFR 自带的 `get_polybasis('line', ...)`，不再手写 Lagrange 公式（与手写版差 ≤1e-15 相对）。
3. 检验脚本（`examples_src/wmles/dev/test_*_dev.py`）只从面通量/RHS 读 τ，列与 H 由脚本按几何独立求出。
4. 去掉对后端私有接口的依赖：采样视图需要已分配的存储，原先手动遍历 `be._extents`/`_primary._pending` 提前分配寄存器库，改为公开的 `be.commit()`；`wmsrc` 用默认 `extent=None`（立即分配）。
5. 采样状态改为每个寄存器库一个二维视图（整列解点 × 变量，按列给行步长，两侧壁面步长正负都已覆盖），外部参数只剩 `uwms, hwm`，取消 `order <= 8` 限制；点采样即权重为 0/1 的列，两种采样共用一条内核路径。
6. 列平均权重改为用 PyFR 线基积分节点基函数，不再解 Vandermonde 方程（9 点时条件数 1.8e6，权重误差 1.4e-10；p=8 内核检验误差 1.2e-9 → 6.3e-13）。

回归：
- 壁模型内核（关闭 debug 后对比）：point、column-mean 的面通量与 RHS 与清理前逐位相同；
- 4 层整算例 t=10 解场：只含第 1 项时与 `s_es400` 逐位相同；含第 2 项时相对差 2e-13（舍入，经混沌动力学放大）；
- level3 熵守恒检验：熵误差、守恒误差仍在 1e-17～1e-15。
- 第 4、5 项：point、column-mean 面通量与 RHS 与改动前逐位相同，4 层整算例 t=10 逐位相同；第 6 项只改变列平均权重的末位（point 路径不受影响）。

## 六、2026-10-26 合入 `liao`：M2.5 TBL 的生产代码（`idea/tavg-pinned`，含 `idea/recycle`）与 `idea/efcount`

从此运行和修改都在本仓库的 `liao` 上进行（环境：`/home/liaofei/workspace/pyfr/activate_pyfr.sh`，唯一的 venv，以可编辑方式安装本仓库）。原来各个 worktree 都已删除，分支保留。

| 改动 | 文件 / 提交 | 说明 | 状态 |
|---|---|---|---|
| 超声速回收入口 `sup-in-recycle` | `navstokes/recycle.py`、`kernels/bcs/sup-in-recycle.mako`、`navstokes/system.py`；`05565bd0`、`c67baae3` | 按回收站重新缩放入口剖面；x-rec 改变时重置平均；日志逐行刷新 | **有效**：M2、M2.5 TBL 正式计算 |
| tavg 不保留累加量的常驻主机拷贝 | `plugins/soln/tavg.py`；`1e42e55c` | 降低主机内存峰值 | **有效**：M2.5，94 万单元 |
| 换传输缓冲区前先释放；锁页申请失败时改用普通内存 | `backends/cuda/base.py`；`1d3c3e89` | 锁页申请曾偶发失败 3 次（CUDAInvalidValue） | **有效** |
| 插件出错时紧急存档 | `integrators/base.py`；`91753c77` | 先让 writer、tavg 做不清零存档，再抛出异常 | **有效** |
| tavg：启动时就申请好传输缓冲区；trigger_write 前先累加到当前时刻；首次调用时设定上一次采样时刻 | `plugins/soln/tavg.py`；`f9dc265f` | 首次调用的改动与 `fix/tavg-first-window` `4b922079` 相同 | **有效**：续算后第一个窗口的 ⟨1⟩ 由 0.998 变为 1；存档时的 ⟨1⟩ 由 139/140 变为 1 |
| 熵滤波动作计数插件 `efcount` | `plugins/soln/efcount.py`；`0719ae55` | 每步读导出量 ef-filter，统计被滤波的单元；只看最后一个子步，所以是下限；自身出错不抛异常 | **工具** |

合并前检验：M2.5 正式配置（开只保正熵滤波）从同一快照算 100 步。合并后的 `liao`（`9d647267`）与原生产代码（`idea/tavg-pinned`）的解场、ef-filter 和各日志都逐位相同（`examples_src/tbl/m2.5/tests/unify/compare.txt`）。
