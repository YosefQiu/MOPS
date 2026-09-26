# PathLine Debug 系统使用指南

## 📖 概述

PathLine debug系统提供了一套可控的调试宏，替代传统的`printf`和`#ifdef`组合。

**优势：**
- ✅ 无需手动注释/删除debug代码
- ✅ 通过编译选项控制开关
- ✅ 可以精确过滤particle和step
- ✅ 生产环境零性能开销
- ✅ 统一的debug输出格式

## 🚀 快速开始

### 1. 默认情况（Debug关闭）

```bash
# 正常编译 - 所有debug宏都被优化掉
bash script/compiler_cuda.sh
```

**结果：** 没有任何debug输出，零性能影响

### 2. 开启Debug输出

编辑 `CMakeLists.txt`，在文件开头添加：

```cmake
# 开启PathLine debug (level 1: 基本输出)
add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
```

重新编译：
```bash
bash script/compiler_cuda.sh
```

**结果：** 会看到PathLine kernel的debug输出

### 3. 精确控制

```cmake
# 只调试particle 0
add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
add_compile_definitions(MOPS_DEBUG_PARTICLE_ID=0)

# 或只调试step 22
add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
add_compile_definitions(MOPS_DEBUG_STEP_ID=22)

# 或两者结合
add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
add_compile_definitions(MOPS_DEBUG_PARTICLE_ID=0)
add_compile_definitions(MOPS_DEBUG_STEP_ID=22)
```

## 📊 Debug级别

```
Level 0: 关闭（生产环境，默认）
Level 1: 基本输出（重要事件）
Level 2: 详细输出（包含中间计算）
Level 3: 非常详细（包含循环内部）
```

示例：
```cmake
# 非常详细的输出
add_compile_definitions(MOPS_DEBUG_PATHLINE=3)
```

## 🔧 可用的Debug宏

### 基本输出

```cpp
// 通用debug (只从thread 0输出)
DEBUG_PATHLINE("PathLine started with %d particles", particle_count);

// 特定particle的debug
DEBUG_PATHLINE_PARTICLE(global_id, "Current depth: %.2f", depth);

// 特定step的debug
DEBUG_PATHLINE_STEP(i_step, "Computing RK4 k1");

// 特定particle在特定step
DEBUG_PATHLINE_PS(global_id, i_step, "Cell: %d, Depth: %.2f", cell_id, depth);
```

### 详细输出

```cpp
// Level 2+ 才输出
DEBUG_PATHLINE_VERBOSE("Intermediate value: %f", value);

// Level 3+ 才输出
DEBUG_PATHLINE_VVERBOSE("Inner loop iteration: %d", i);
```

### 工具宏

```cpp
// 输出vec3
vec3 pos = ...;
DEBUG_PATHLINE_VEC3(global_id, "Position", pos);
// 输出: [PathLine P0] Position: (123.456, 789.012, 345.678)

// 输出数组
double weights[3] = {0.3, 0.5, 0.2};
DEBUG_PATHLINE_ARRAY(global_id, "Weights", weights, 3);
// 输出: [PathLine P0] Weights: [0.300000, 0.500000, 0.200000]

// 条件检查
DEBUG_PATHLINE_CHECK(cell_id >= 0, global_id, "Invalid cell: %d", cell_id);
// 只在cell_id < 0时输出错误信息
```

## 💡 实际使用示例

### 场景1: 调试特定粒子的轨迹

```cmake
# CMakeLists.txt
add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
add_compile_definitions(MOPS_DEBUG_PARTICLE_ID=0)
```

```bash
bash script/compiler_cuda.sh
python tutorial/your_test.py
```

**输出示例：**
```
[PathLine P0] Starting position: (6.371e6, 0.0, 0.0)
[PathLine P0] Current depth: 100.00
[PathLine P0] Cell: 68814, Layer: 5
...
```

### 场景2: 调试特定时间步

```cmake
# CMakeLists.txt
add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
add_compile_definitions(MOPS_DEBUG_STEP_ID=22)
```

**用途：** 当你发现"step 22有问题"时，只查看step 22的输出

### 场景3: 调试特定事件

```cmake
# CMakeLists.txt
add_compile_definitions(MOPS_DEBUG_PATHLINE=2)  # Verbose
add_compile_definitions(MOPS_DEBUG_PARTICLE_ID=0)
add_compile_definitions(MOPS_DEBUG_STEP_ID=22)
```

**用途：** 深入调试particle 0在step 22的所有细节

### 场景4: 生产运行

```cmake
# CMakeLists.txt
# 不添加任何MOPS_DEBUG_PATHLINE定义
# 或显式设置为0
add_compile_definitions(MOPS_DEBUG_PATHLINE=0)
```

**结果：** 所有debug代码被编译器完全优化掉，零开销

## 🎯 替换前后对比

### 之前（需要手动注释）

```cpp
#ifdef __CUDA_ARCH__
if (global_id == 0) {
    printf("Step %d: cell=%d, depth=%.2f\n", i_step, cell_id, depth);
}
#endif

// 要关闭debug？手动注释：
// #ifdef __CUDA_ARCH__
// if (global_id == 0) {
//     printf("Step %d: cell=%d, depth=%.2f\n", i_step, cell_id, depth);
// }
// #endif
```

### 现在（通过编译选项控制）

```cpp
DEBUG_PATHLINE_PS(global_id, i_step, "cell=%d, depth=%.2f", cell_id, depth);

// 要关闭debug？在CMakeLists.txt中设置：
// add_compile_definitions(MOPS_DEBUG_PATHLINE=0)
// 或者直接删除该定义行
```

## 📝 最佳实践

1. **开发阶段：** 使用Level 1或2，只调试关注的particle/step
   ```cmake
   add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
   add_compile_definitions(MOPS_DEBUG_PARTICLE_ID=0)
   ```

2. **深入调试：** 使用Level 3，获取所有细节
   ```cmake
   add_compile_definitions(MOPS_DEBUG_PATHLINE=3)
   add_compile_definitions(MOPS_DEBUG_PARTICLE_ID=0)
   add_compile_definitions(MOPS_DEBUG_STEP_ID=22)
   ```

3. **提交代码前：** 确保关闭debug
   ```cmake
   # 删除或注释掉所有MOPS_DEBUG_PATHLINE相关定义
   ```

4. **性能测试：** 务必关闭debug
   ```cmake
   add_compile_definitions(MOPS_DEBUG_PATHLINE=0)
   ```

## ⚠️ 注意事项

1. **性能影响：** Debug开启时会显著降低性能（大量printf）
   - Level 1: ~10-20% 性能损失
   - Level 2: ~30-50% 性能损失
   - Level 3: ~50-80% 性能损失

2. **输出量：** 调试多个粒子会产生海量输出
   - 建议只调试1个particle
   - 或使用step过滤减少输出

3. **编译时间：** Debug开启不会增加编译时间（宏在编译期展开）

4. **库大小：** Debug关闭时，库大小不变（代码被优化掉）

## 🔍 常见用例

### 用例1: 查找NaN的来源

```cpp
DEBUG_PATHLINE_CHECK(!isnan(velocity.x()), global_id, 
                     "NaN velocity at step %d", i_step);
DEBUG_PATHLINE_VEC3(global_id, "Velocity before RK4", velocity);
```

### 用例2: 跟踪粒子轨迹

```cpp
DEBUG_PATHLINE_PS(global_id, i_step, 
                 "Position: (%.2f,%.2f,%.2f), Cell: %d", 
                 pos.x(), pos.y(), pos.z(), cell_id);
```

### 用例3: 验证边界处理

```cpp
DEBUG_PATHLINE_PARTICLE(global_id, 
                       "Hit boundary: depth=%.2f, max_depth=%.2f", 
                       depth, max_depth);
```

## 📚 更多信息

- Debug宏定义：`PathLineDebug.cuh`
- 使用示例：查看 `MPASOPathLineKernel.cu` 中的实际用法
- 问题反馈：在代码中搜索 `DEBUG_PATHLINE` 查看所有使用位置

---

**Created:** 2026-09-21  
**Updated:** 2026-09-21
