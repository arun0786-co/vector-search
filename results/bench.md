| index | build (s) | index size (MB) | queries/s at recall >= 0.9 | queries/s at recall >= 0.95 | queries/s at recall >= 0.99 |
|---|---|---|---|---|---|
| ivf nlist=1024 | 32 | 123 | 5,264 | 2,473 | 1,378 |
| ivfpq nlist=1024 m=5 rerank=0 | 48 | 11 | not reached | not reached | not reached |
| ivfpq nlist=1024 m=25 rerank=0 | 61 | 34 | 2,801 | 1,932 | not reached |
| ivfpq nlist=1024 m=5 rerank=100 | 46 | 11 | 4,078 | not reached | not reached |
| hnsw M=8 efC=100 | 107 | 204 | 7,390 | 3,875 | 1,959 |
| hnsw M=16 efC=200 | 283 | 279 | 16,581 | 9,166 | 5,019 |
| exact | 0 | 118 | 112 (recall 1.0) | | |
