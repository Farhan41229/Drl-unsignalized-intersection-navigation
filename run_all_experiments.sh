#!/bin/bash
set -e
cd Original && bash run_pipeline.sh && cd ..
cd B && bash run_pipeline.sh && cd ..
