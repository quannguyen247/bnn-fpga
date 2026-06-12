`ifndef BNN_DEFS
`define BNN_DEFS

// ============================================================
//  BNN Network Topology
// ============================================================
`define BNN_IN_W      784     // 28x28 binary pixels
`define BNN_FC1_N     128     // FC Layer 1: 784 -> 128
`define BNN_FC2_N      64     // FC Layer 2: 128 -> 64
`define BNN_FC3_N      10     // FC Layer 3:  64 -> 10
`define BNN_CLASS_W     4     // Output class index width

`endif
