`ifndef BNN_DEFS
`define BNN_DEFS

// ============================================================
//  BNN Network Topology & Processing Config
//  Topology: 784 → 64 → 32 → 10  (sweet spot accuracy/resource)
//  Processing: chunk-serial, CHUNK_W bits moi cycle
//
//  Resource uoc tinh (Artix-7 xc7a35t):
//    LUT ~4,000 / 20,800 (19%)
//    FF  ~2,000 / 41,600 (5%)
//  Latency: 32 cycles / inference
// ============================================================
`define BNN_IN_W      784     // 28x28 binary pixels
`define BNN_FC1_N      64     // FC Layer 1: 784 -> 64
`define BNN_FC2_N      32     // FC Layer 2:  64 -> 32
`define BNN_FC3_N      10     // FC Layer 3:  32 -> 10
`define BNN_CHUNK_W    32     // Bits xu ly moi cycle (serial)
`define BNN_CLASS_W     4     // ceil(log2(10)) = 4

`endif
