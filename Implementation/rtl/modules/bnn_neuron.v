`timescale 1ns / 1ps

// ============================================================
//  bnn_neuron — XNOR-Popcount Binary Neuron
//  Tinh so bit giong nhau giua data va weight (binary dot-product)
//  Output: unsigned popcount ∈ [0, N]
//  Equivalent: (2 × popcount − N) cho signed inner product
// ============================================================
module bnn_neuron #(
    parameter N  = 64,
    parameter RW = $clog2(N + 1)        // Result width: du cho 0..N
)(
    input  wire [N-1:0]  data,
    input  wire [N-1:0]  weight,
    output wire [RW-1:0] popcount
);

    // XNOR: bit = 1 khi data == weight
    wire [N-1:0] xnor_out = data ~^ weight;

    // Popcount: loop → Vivado synthesize thanh adder tree
    reg [RW-1:0] cnt;
    integer j;
    always @(*) begin
        cnt = {RW{1'b0}};
        for (j = 0; j < N; j = j + 1)
            cnt = cnt + xnor_out[j];
    end

    assign popcount = cnt;

endmodule
