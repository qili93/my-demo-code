#pragma once

#include "utility.h"

class OMModelBuild {
public:
  OMModelBuild() {}
  ~OMModelBuild() {}

  bool GenGraph(ge::Graph& graph);
  bool SaveModel(ge::Graph& om_graph, std::string model_path);
};