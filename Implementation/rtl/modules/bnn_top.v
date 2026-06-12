`timescale 1ns / 1ps
`include "bnn_defs.vh"

// ============================================================
//  bnn_top — 3-stage Pipelined Binary Neural Network
//  Pipeline: FC1+BN1 | FC2+BN2 | FC3+Argmax
//  Latency : 3 cycles
//  Throughput: 1 inference / cycle (sau pipeline fill)
// ============================================================
module bnn_top (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [`BNN_IN_W-1:0]    input_data,
    input  wire                     input_valid,
    output reg  [`BNN_CLASS_W-1:0] output_class,
    output reg                      output_valid
);

    // ---- Popcount result widths ----
    localparam RW1 = $clog2(`BNN_IN_W  + 1);   // FC1: 10 bits
    localparam RW2 = $clog2(`BNN_FC1_N + 1);   // FC2:  8 bits
    localparam RW3 = $clog2(`BNN_FC2_N + 1);   // FC3:  7 bits

    // ============================================================
    //  Stage 0: input → FC1 → BN1 (combinational)
    // ============================================================
    wire [`BNN_FC1_N*RW1-1:0] fc1_out;
    bnn_fc_layer #(
        .N_IN (`BNN_IN_W),
        .N_OUT(`BNN_FC1_N),
        .SEED (32'hDEAD_BEEF)
    ) u_fc1 (.data(input_data), .result(fc1_out));

    wire [`BNN_FC1_N-1:0] bn1_out;
    bnn_batchnorm #(
        .N_CH    (`BNN_FC1_N),
        .RESULT_W(RW1),
        .N_IN    (`BNN_IN_W)
    ) u_bn1 (.data(fc1_out), .act(bn1_out));

    // ---- Pipeline register: Stage 0 → 1 ----
    reg [`BNN_FC1_N-1:0] s1_act;
    reg                   s1_valid;

    // ============================================================
    //  Stage 1: s1_act → FC2 → BN2 (combinational)
    // ============================================================
    wire [`BNN_FC2_N*RW2-1:0] fc2_out;
    bnn_fc_layer #(
        .N_IN (`BNN_FC1_N),
        .N_OUT(`BNN_FC2_N),
        .SEED (32'hCAFE_BABE)
    ) u_fc2 (.data(s1_act), .result(fc2_out));

    wire [`BNN_FC2_N-1:0] bn2_out;
    bnn_batchnorm #(
        .N_CH    (`BNN_FC2_N),
        .RESULT_W(RW2),
        .N_IN    (`BNN_FC1_N)
    ) u_bn2 (.data(fc2_out), .act(bn2_out));

    // ---- Pipeline register: Stage 1 → 2 ----
    reg [`BNN_FC2_N-1:0] s2_act;
    reg                   s2_valid;

    // ============================================================
    //  Stage 2: s2_act → FC3 → Argmax (combinational)
    // ============================================================
    wire [`BNN_FC3_N*RW3-1:0] fc3_out;
    bnn_fc_layer #(
        .N_IN (`BNN_FC2_N),
        .N_OUT(`BNN_FC3_N),
        .SEED (32'h1234_5678)
    ) u_fc3 (.data(s2_act), .result(fc3_out));

    wire [`BNN_CLASS_W-1:0] argmax_out;
    bnn_argmax #(
        .N_IN  (`BNN_FC3_N),
        .DATA_W(RW3)
    ) u_argmax (.data(fc3_out), .idx(argmax_out));

    // ============================================================
    //  Pipeline Registers
    // ============================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            s1_valid     <= 1'b0;
            s1_act       <= {`BNN_FC1_N{1'b0}};
            s2_valid     <= 1'b0;
            s2_act       <= {`BNN_FC2_N{1'b0}};
            output_valid <= 1'b0;
            output_class <= {`BNN_CLASS_W{1'b0}};
        end else begin
            // Stage 0 → 1
            s1_valid     <= input_valid;
            s1_act       <= bn1_out;
            // Stage 1 → 2
            s2_valid     <= s1_valid;
            s2_act       <= bn2_out;
            // Stage 2 → output
            output_valid <= s2_valid;
            output_class <= argmax_out;
        end
    end

endmodule
