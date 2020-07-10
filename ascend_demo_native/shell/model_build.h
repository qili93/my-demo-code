#pragma once

#include "graph/graph.h"
#include "utils.h"
#include "logging.h"

class OMModelBuild {
public:
  OMModelBuild() {}
  ~OMModelBuild() {}

  bool GenGraph(ge::Graph& graph);
  bool SaveModel(ge::Graph& om_graph, std::string model_path);
};