`timescale 1ns / 1ps

// ============================================================
//  bnn_fc_layer — Binary Fully-Connected Layer
//  N_OUT neurons, moi neuron: XNOR-popcount(data, weight)
//  Weights khoi tao = xorshift32 PRNG (demo).
//  Thuc te: thay the bang trained weights (load tu file/AXI).
// ============================================================
module bnn_fc_layer #(
    parameter N_IN  = 64,
    parameter N_OUT = 32,
    parameter RW    = $clog2(N_IN + 1),
    parameter SEED  = 32'hDEAD_BEEF
)(
    input  wire [N_IN-1:0]      data,
    output wire [N_OUT*RW-1:0]  result
);

    // ---- Weight storage (ROM sau synthesis) ----
    reg [N_IN-1:0] weights [0:N_OUT-1];

    // Khoi tao bang xorshift32 PRNG — deterministic, ~50% ones
    integer i, j;
    reg [31:0] seed;
    initial begin
        seed = SEED;
        for (i = 0; i < N_OUT; i = i + 1)
            for (j = 0; j < N_IN; j = j + 1) begin
                seed = seed ^ (seed << 13);
                seed = seed ^ (seed >> 17);
                seed = seed ^ (seed << 5);
                weights[i][j] = seed[0];
            end
    end

    // ---- Instantiate neurons ----
    genvar g;
    generate
        for (g = 0; g < N_OUT; g = g + 1) begin : gen_neuron
            wire [N_IN-1:0] w = weights[g];
            bnn_neuron #(.N(N_IN)) u_neuron (
                .data    (data),
                .weight  (w),
                .popcount(result[g*RW +: RW])
            );
        end
    endgenerate

endmodule
