#!/bin/bash
nvcc -O3 -maxrregcount=32 ex2-v2.cu -o ex2
