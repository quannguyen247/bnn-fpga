`timescale 1ns / 1ps

// ============================================================
//  bnn_batchnorm — Batch Normalization + Sign Activation
//  So sanh popcount voi threshold → output binary {0, 1}
//  Demo: threshold = N_IN/2 (majority vote)
//  Thuc te: threshold tinh tu trained BN params (γ, β, μ, σ)
// ============================================================
module bnn_batchnorm #(
    parameter N_CH     = 128,           // So channels (neurons)
    parameter RESULT_W = 10,            // Bit-width cua popcount input
    parameter N_IN     = 784            // Input dim cua FC layer truoc
)(
    input  wire [N_CH*RESULT_W-1:0] data,
    output wire [N_CH-1:0]          act
);

    // Threshold: popcount > N_IN/2 → activation = 1
    localparam [RESULT_W-1:0] THRESH = N_IN / 2;

    genvar g;
    generate
        for (g = 0; g < N_CH; g = g + 1) begin : gen_bn
            assign act[g] = (data[g*RESULT_W +: RESULT_W] > THRESH);
        end
    endgenerate

endmodule
