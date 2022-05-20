## 修改Fluid模型输出

第一步：修改Paddle Python代码

```bash
# 找到paddle安装目录
pip show paddlepaddle
# ~/Documents/miniconda3/envs/py37env/lib/python3.7/site-packages/paddle

# 修改如下文件
site-packages/paddle/fluid/io.py
# 注释如下代码：
    # fix the bug that the activation op's output as target will be pruned.
    # will affect the inference performance.
    # TODO(Superjomn) add an IR pass to remove 1-scale op.
    # with program_guard(main_program):
    #     uniq_target_vars = []
    #     for i, var in enumerate(target_vars):
    #         if isinstance(var, Variable):
    #             var = layers.scale(
    #                 var, 1., name="save_infer_model/scale_{}".format(i))
    #         uniq_target_vars.append(var)
    #     target_vars = uniq_target_vars
    target_var_name_list = [var.name for var in target_vars]
```

第二步：修改fluid_clip.py文件中新的输入和输出

```python
feed_target_names = ["VISface_landmark_0"]
fetch_targets = [test_program.current_block().var("VISface_landmark_22968.avg_pool.output.1.tmp_0")]
```

第三步：运行python fluid_clip.py即可得到裁剪后fluid模型