import paddle

paddle.enable_static()

# 定义网络结构，该任务中使用线性回归模型，网络由一个fc层构成
def linear_regression_net(input, hidden):
    out = paddle.static.nn.fc(input, hidden)
    return out

#main_program = paddle.static.Program()
#startup_program = paddle.static.Program()
main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()

x = paddle.static.data(name='x', shape=[None, 13], dtype='float32')
y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')

batch_size = 20
train_reader = paddle.io.DataLoader(paddle.text.datasets.UCIHousing(mode='train'), feed_list=[x, y], batch_size=batch_size, shuffle=True)
test_reader = paddle.io.DataLoader(paddle.text.datasets.UCIHousing(mode='test'), feed_list=[x], batch_size=batch_size)

# 调用网络，执行前向计算
prediction = linear_regression_net(x, 1)

# 计算损失值
loss = paddle.fluid.layers.cross_entropy(input=prediction, label=y)
avg_loss = paddle.mean(loss)

# 定义优化器，并调用minimize接口计算和更新梯度
sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

# 此处创建执行器
exe = paddle.static.Executor(place=paddle.device.IPUPlace())
exe.run(startup_program)

# IPU配置
ipu_strategy = paddle.static.IpuStrategy()
ipu_strategy.set_graph_config(is_training=True)
ipu_strategy.set_pipelining_config(batches_per_step=batch_size)
print(f"start compiling model for ipu, it will need some minutes")
ipu_compiler = paddle.static.IpuCompiledProgram(main_program, ipu_strategy=ipu_strategy)
main_program = ipu_compiler.compile(feed_list=[x.name, y.name], fetch_list=[avg_loss.name])
print(f"finish model compiling!")

max_epoch_num = 100  # 执行max_epoch_num次训练
for epoch in range(max_epoch_num):
    for batch_id, (x_tensor, y_tensor) in enumerate(train_reader()):
        avg_loss_value, = exe.run(program=main_program, feed={'x': x_tensor, 'y': y_tensor}, fetch_list=[avg_loss])
        if batch_id % 10 == 0 and batch_id is not 0:
            print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss_value))

# 静态图中需要使用save_inference_model来保存模型，以供预测使用
ipu_compiler._backend.weights_to_host()
paddle.static.save_inference_model('./static_linear', [x], [prediction], exe)

# 执行预测
infer_exe = paddle.static.Executor()
inference_scope = paddle.static.Scope()
# 使用训练好的模型做预测
with paddle.static.scope_guard(inference_scope):
    # 静态图中需要使用load_inference_model来加载之前保存的模型
    [inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model('./static_linear', infer_exe)

    # 读取一组测试数据
    (infer_x, infer_y) = next(test_reader())

    # 静态图中预测时也需要调用执行器的run方法执行计算过程，并指定之前加载的inference_program
    results = infer_exe.run(
       inference_program,
       feed={feed_target_names[0]: infer_x},
       fetch_list=fetch_targets)

    print("id: prediction ground_truth")
    for idx, val in enumerate(results[0]):
       print("%d: %.2f %.2f" % (idx, val, infer_y.__array__()[idx]))


