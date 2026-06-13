`timescale 1ns / 1ps
`include "bnn_defs.vh"

// ============================================================
//  bnn_top — FSM-Controlled Chunk-Serial BNN
//
//  Kien truc moi: thay vi pipeline 3-stage (1 cycle/stage),
//  dung FSM dieu khien 3 FC layers chay TUAN TU.
//  Moi FC layer xu ly CHUNK_W bits/cycle (serial).
//
//  FSM: IDLE → FC1 (26 cy) → FC2 (3 cy) → FC3 (2 cy) → output
//  Tong latency: 32 cycles / inference
//  BN1, BN2, Argmax = combinational (khong can state rieng)
//
//  Uu diem so voi thiet ke cu:
//    - LUT giam (~3K thay vi ~25K+) vi popcount chi 32-bit
//    - FF tang (~2K) vi dung accumulator + data register
//    - Critical path ngan → timing de dat 200MHz
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
    localparam RW2 = $clog2(`BNN_FC1_N + 1);   // FC2:  7 bits
    localparam RW3 = $clog2(`BNN_FC2_N + 1);   // FC3:  6 bits

    // ============================================================
    //  FSM States
    // ============================================================
    localparam S_IDLE = 2'd0;
    localparam S_FC1  = 2'd1;
    localparam S_FC2  = 2'd2;
    localparam S_FC3  = 2'd3;

    reg [1:0] state;

    // ---- FC layer control signals ----
    wire fc1_done, fc2_done, fc3_done;

    // Start signals (Mealy output — combinational, 0 cycle overhead)
    wire fc1_start = (state == S_IDLE) & input_valid;
    wire fc2_start = (state == S_FC1)  & fc1_done;
    wire fc3_start = (state == S_FC2)  & fc2_done;

    // ---- FC layer result buses ----
    wire [`BNN_FC1_N*RW1-1:0] fc1_result;
    wire [`BNN_FC2_N*RW2-1:0] fc2_result;
    wire [`BNN_FC3_N*RW3-1:0] fc3_result;

    // ---- BN activation outputs ----
    wire [`BNN_FC1_N-1:0] bn1_out;
    wire [`BNN_FC2_N-1:0] bn2_out;

    // ---- Argmax output ----
    wire [`BNN_CLASS_W-1:0] argmax_out;

    // ============================================================
    //  FC Layer 1: input_data (784b) → 64 neurons
    //  Latency: ceil(784/32)+1 = 26 cycles
    // ============================================================
    bnn_fc_layer #(
        .N_IN   (`BNN_IN_W),
        .N_OUT  (`BNN_FC1_N),
        .CHUNK_W(`BNN_CHUNK_W),
        .SEED   (32'hDEAD_BEEF)
    ) u_fc1 (
        .clk   (clk),
        .rst_n (rst_n),
        .start (fc1_start),
        .data  (input_data),
        .result(fc1_result),
        .done  (fc1_done)
    );

    // ---- Batch Norm 1 (combinational) ----
    // Khi fc1_done, accumulators stable → bn1_out valid
    bnn_batchnorm #(
        .N_CH    (`BNN_FC1_N),
        .RESULT_W(RW1),
        .N_IN    (`BNN_IN_W)
    ) u_bn1 (
        .data(fc1_result),
        .act (bn1_out)
    );

    // ============================================================
    //  FC Layer 2: bn1_out (64b) → 32 neurons
    //  Latency: ceil(64/32)+1 = 3 cycles
    // ============================================================
    bnn_fc_layer #(
        .N_IN   (`BNN_FC1_N),
        .N_OUT  (`BNN_FC2_N),
        .CHUNK_W(`BNN_CHUNK_W),
        .SEED   (32'hCAFE_BABE)
    ) u_fc2 (
        .clk   (clk),
        .rst_n (rst_n),
        .start (fc2_start),
        .data  (bn1_out),
        .result(fc2_result),
        .done  (fc2_done)
    );

    // ---- Batch Norm 2 (combinational) ----
    bnn_batchnorm #(
        .N_CH    (`BNN_FC2_N),
        .RESULT_W(RW2),
        .N_IN    (`BNN_FC1_N)
    ) u_bn2 (
        .data(fc2_result),
        .act (bn2_out)
    );

    // ============================================================
    //  FC Layer 3: bn2_out (32b) → 10 neurons (output layer)
    //  Latency: ceil(32/32)+1 = 2 cycles
    // ============================================================
    bnn_fc_layer #(
        .N_IN   (`BNN_FC2_N),
        .N_OUT  (`BNN_FC3_N),
        .CHUNK_W(`BNN_CHUNK_W),
        .SEED   (32'h1234_5678)
    ) u_fc3 (
        .clk   (clk),
        .rst_n (rst_n),
        .start (fc3_start),
        .data  (bn2_out),
        .result(fc3_result),
        .done  (fc3_done)
    );

    // ---- Argmax (combinational) ----
    // Tim neuron co popcount cao nhat → class index
    bnn_argmax #(
        .N_IN  (`BNN_FC3_N),
        .DATA_W(RW3)
    ) u_argmax (
        .data(fc3_result),
        .idx (argmax_out)
    );

    // ============================================================
    //  FSM + Output Register
    // ============================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            output_valid <= 1'b0;
            output_class <= {`BNN_CLASS_W{1'b0}};
        end else begin
            output_valid <= 1'b0;               // Default: 1-cycle pulse
            case (state)
                S_IDLE: begin
                    if (input_valid)
                        state <= S_FC1;
                end
                S_FC1: begin
                    if (fc1_done)
                        state <= S_FC2;         // BN1 valid → FC2 starts
                end
                S_FC2: begin
                    if (fc2_done)
                        state <= S_FC3;         // BN2 valid → FC3 starts
                end
                S_FC3: begin
                    if (fc3_done) begin
                        state        <= S_IDLE;
                        output_valid <= 1'b1;
                        output_class <= argmax_out;
                    end
                end
            endcase
        end
    end

endmodule
