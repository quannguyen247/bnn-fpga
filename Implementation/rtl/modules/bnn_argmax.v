`timescale 1ns / 1ps

// ============================================================
//  bnn_argmax — Tim index co gia tri lon nhat (Tree-based)
//  Priority: index thap uu tien khi gia tri bang nhau
//
//  Redesigned as a balanced binary tree of depth 4 to reduce
//  logic depth from 17 LUTs to ~4-5 LUTs. Meets 200MHz timing.
// ============================================================
module bnn_argmax #(
    parameter N_IN   = 10,
    parameter DATA_W = 7
)(
    input  wire [N_IN*DATA_W-1:0]  data,
    output wire [$clog2(N_IN)-1:0] idx
);

    localparam IDX_W = $clog2(N_IN);

    // Unpack input data into array, pad missing slots with 0s
    wire [DATA_W-1:0] val [0:9];
    genvar g;
    generate
        for (g = 0; g < 10; g = g + 1) begin : gen_val
            if (g < N_IN)
                assign val[g] = data[g*DATA_W +: DATA_W];
            else
                assign val[g] = {DATA_W{1'b0}};
        end
    endgenerate

    // ------------------------------------------------------------
    //  Tree Level 1: 10 inputs -> 5 nodes
    // ------------------------------------------------------------
    wire [DATA_W-1:0] v1 [0:4];
    wire [IDX_W-1:0]  i1 [0:4];

    assign v1[0] = (val[0] >= val[1]) ? val[0] : val[1];
    assign i1[0] = (val[0] >= val[1]) ? 4'd0   : 4'd1;

    assign v1[1] = (val[2] >= val[3]) ? val[2] : val[3];
    assign i1[1] = (val[2] >= val[3]) ? 4'd2   : 4'd3;

    assign v1[2] = (val[4] >= val[5]) ? val[4] : val[5];
    assign i1[2] = (val[4] >= val[5]) ? 4'd4   : 4'd5;

    assign v1[3] = (val[6] >= val[7]) ? val[6] : val[7];
    assign i1[3] = (val[6] >= val[7]) ? 4'd6   : 4'd7;

    assign v1[4] = (val[8] >= val[9]) ? val[8] : val[9];
    assign i1[4] = (val[8] >= val[9]) ? 4'd8   : 4'd9;

    // ------------------------------------------------------------
    //  Tree Level 2: 5 nodes -> 3 nodes
    // ------------------------------------------------------------
    wire [DATA_W-1:0] v2 [0:2];
    wire [IDX_W-1:0]  i2 [0:2];

    assign v2[0] = (v1[0] >= v1[1]) ? v1[0] : v1[1];
    assign i2[0] = (v1[0] >= v1[1]) ? i1[0] : i1[1];

    assign v2[1] = (v1[2] >= v1[3]) ? v1[2] : v1[3];
    assign i2[1] = (v1[2] >= v1[3]) ? i1[2] : i1[3];

    // Node 4 passes through
    assign v2[2] = v1[4];
    assign i2[2] = i1[4];

    // ------------------------------------------------------------
    //  Tree Level 3: 3 nodes -> 2 nodes
    // ------------------------------------------------------------
    wire [DATA_W-1:0] v3 [0:1];
    wire [IDX_W-1:0]  i3 [0:1];

    assign v3[0] = (v2[0] >= v2[1]) ? v2[0] : v2[1];
    assign i3[0] = (v2[0] >= v2[1]) ? i2[0] : i2[1];

    // Node 2 passes through
    assign v3[1] = v2[2];
    assign i3[1] = i2[2];

    // ------------------------------------------------------------
    //  Tree Level 4: 2 nodes -> 1 output
    // ------------------------------------------------------------
    wire [DATA_W-1:0] v4;
    wire [IDX_W-1:0]  i4;

    assign v4 = (v3[0] >= v3[1]) ? v3[0] : v3[1];
    assign i4 = (v3[0] >= v3[1]) ? i3[0] : i3[1];

    assign idx = i4;

endmodule
