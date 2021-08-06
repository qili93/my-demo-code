#include "utility.h"


/*
* 对op名称添加后缀，保证op名称不重复
*/
const char *GetGlobalIndex( const char *str ) {
    static std::string strIndex;
    static uint32_t globalIndex = 0;
    strIndex.erase();
    strIndex.append(str);
    strIndex.append("_" + to_string(globalIndex));
    globalIndex++;
    return strIndex.c_str();
}

bool GenGraph(ge::Graph& graph) {
  // input x: A 4D tensor of input images - bs, ic, h, w
  ge::TensorDesc input_desc_x(ge::Shape({ 1L, 2L, 2L, 2L }), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_x = ge::op::Data("input_x");//.set_attr_index(0);
  input_x.update_input_desc_x(input_desc_x);
  input_x.update_output_desc_y(input_desc_x);

  // input bias
  ge::TensorDesc input_desc_bias(ge::Shape({ 2L }), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_bias = ge::op::Data("input_bias");//.set_attr_index(0);
  input_bias.update_input_desc_x(input_desc_bias);
  input_bias.update_output_desc_y(input_desc_bias);

  // input mean
  ge::TensorDesc input_desc_mean(ge::Shape({ 2L }), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_mean = ge::op::Data("input_mean");//.set_attr_index(0);
  input_mean.update_input_desc_x(input_desc_mean);
  input_mean.update_output_desc_y(input_desc_mean);

  // input scale
  ge::TensorDesc input_desc_scale(ge::Shape({ 2L }), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_scale = ge::op::Data("input_scale");//.set_attr_index(0);
  input_scale.update_input_desc_x(input_desc_scale);
  input_scale.update_output_desc_y(input_desc_scale);

  // input variance
  ge::TensorDesc input_desc_variance(ge::Shape({ 2L }), ge::FORMAT_ND, ge::DT_FLOAT);
  auto input_variance = ge::op::Data("input_variance");//.set_attr_index(0);
  input_variance.update_input_desc_x(input_desc_variance);
  input_variance.update_output_desc_y(input_desc_variance);

  // batch norm op
  auto batch_norm_op = ge::op::BatchNorm("y__1");
  batch_norm_op.set_input_x(input_x);
  batch_norm_op.set_input_scale(input_scale);
  batch_norm_op.set_input_offset(input_bias);
  batch_norm_op.set_input_mean(input_mean);
  batch_norm_op.set_input_variance(input_variance);
  batch_norm_op.set_attr_epsilon(0);
  batch_norm_op.set_attr_data_format("NCHW");
  batch_norm_op.set_attr_is_training(false);
  TENSOR_UPDATE_INPUT(batch_norm_op, x, ge::FORMAT_NCHW, ge::DT_FLOAT);
  TENSOR_UPDATE_INPUT(batch_norm_op, scale, ge::FORMAT_NCHW, ge::DT_FLOAT);
  TENSOR_UPDATE_INPUT(batch_norm_op, offset, ge::FORMAT_NCHW, ge::DT_FLOAT);
  TENSOR_UPDATE_INPUT(batch_norm_op, mean, ge::FORMAT_NCHW, ge::DT_FLOAT);
  TENSOR_UPDATE_INPUT(batch_norm_op, variance, ge::FORMAT_NCHW, ge::DT_FLOAT);
  TENSOR_UPDATE_OUTPUT(batch_norm_op, y, ge::FORMAT_NCHW, ge::DT_FLOAT);

  // Const Tensor of new shape
  auto new_shape = ge::Shape( {2L} );
  ge::TensorDesc new_shape_desc( new_shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::Tensor new_shape_tensor( new_shape_desc );
  uint32_t out_size_length = new_shape.GetShapeSize();
  int *pdata = new( std::nothrow ) int[out_size_length];
  pdata[0] = 2;
  pdata[1] = 4;
  ATC_CALL(new_shape_tensor.SetData( reinterpret_cast<uint8_t *>( pdata ), out_size_length * sizeof(int) ));
  // Const OP of new shape
  auto new_shape_op = ge::op::Const( GetGlobalIndex( "Const/new_size") );
  new_shape_op.set_attr_value( new_shape_tensor );
  // TENSOR_UPDATE_OUTPUT(new_shape_op, y, ge::FORMAT_ND, ge::DT_INT32);

  // Reshape OP
  auto reshape_op = ge::op::Reshape( GetGlobalIndex( "Reshape") );
  //reshape_op.set_input_x(batch_norm_op, "y");
  reshape_op.set_input_x(input_x);
  reshape_op.set_input_shape(new_shape_op);
  TENSOR_UPDATE_INPUT(reshape_op, x, ge::FORMAT_ND, ge::DT_FLOAT);
  TENSOR_UPDATE_INPUT(reshape_op, shape, ge::FORMAT_ND, ge::DT_INT32);
  TENSOR_UPDATE_OUTPUT(reshape_op, y, ge::FORMAT_ND, ge::DT_FLOAT);

  // Data OP
  auto data_op = ge::op::Identity(GetGlobalIndex( "Identity"));
  data_op.set_input_x(batch_norm_op, "batch_mean");
  TENSOR_UPDATE_INPUT(data_op, x, ge::FORMAT_NCHW, ge::DT_FLOAT);
  TENSOR_UPDATE_OUTPUT(data_op, y, ge::FORMAT_NCHW, ge::DT_FLOAT);

  // Set inputs and outputs
  std::vector<ge::Operator> input_nodes{ input_x, input_scale, input_bias, input_mean, input_variance };
  // std::vector<ge::Operator> outputs{ net_reshape2 };
  std::vector<ge::Operator> output_nodes{ reshape_op, data_op };
  // set input node attr index is node size > 1
  if (input_nodes.size() > 1) {
    int idx = 0;
    for (auto node : input_nodes) {
      node.SetAttr("index", idx);
      idx++;
    }
  }
  graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  return true;
}