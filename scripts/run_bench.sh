#!/bin/sh
# The full sweep behind results/bench.csv. Takes about half an hour on an M4 Pro.
set -e
D=${1:-data/glove-25}
OUT=${2:-results/bench.csv}
echo "index,params,build_s,index_mb,setting,qps,recall10" > "$OUT"
./build/bench "$D" exact 1000 >> "$OUT"
./build/bench "$D" ivf 1024 1,2,4,8,16,32,64,128 >> "$OUT"
./build/bench "$D" ivfpq 1024 5 0 1,2,4,8,16,32,64,128 >> "$OUT"
./build/bench "$D" ivfpq 1024 25 0 1,2,4,8,16,32,64,128 >> "$OUT"
./build/bench "$D" ivfpq 1024 5 100 1,2,4,8,16,32,64,128 >> "$OUT"
./build/bench "$D" hnsw 8 100 10,20,40,80,160,320,640 >> "$OUT"
./build/bench "$D" hnsw 16 200 10,20,40,80,160,320,640 >> "$OUT"
