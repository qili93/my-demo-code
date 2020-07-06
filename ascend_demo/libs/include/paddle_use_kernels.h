#pragma once
#include "paddle_lite_factory_helper.h"

USE_LITE_KERNEL(unsqueeze, kHost, kAny, kAny, def);
USE_LITE_KERNEL(unsqueeze2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(expand, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(subgraph, kASCEND, kAny, kNCHW, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);
USE_LITE_KERNEL(squeeze, kHost, kAny, kAny, def);
USE_LITE_KERNEL(squeeze2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(multiclass_nms, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms2, kHost, kFloat, kNCHW, def);