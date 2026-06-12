`timescale 1ns / 1ps

// ============================================================
//  bnn_argmax — Tim index co gia tri lon nhat
//  Priority: index thap uu tien khi gia tri bang nhau
// ============================================================
module bnn_argmax #(
    parameter N_IN   = 10,
    parameter DATA_W = 7
)(
    input  wire [N_IN*DATA_W-1:0]  data,
    output wire [$clog2(N_IN)-1:0] idx
);

    localparam IDX_W = $clog2(N_IN);

    reg [IDX_W-1:0]  best_idx;
    reg [DATA_W-1:0] best_val;
    integer i;

    always @(*) begin
        best_val = data[DATA_W-1:0];
        best_idx = {IDX_W{1'b0}};
        for (i = 1; i < N_IN; i = i + 1)
            if (data[i*DATA_W +: DATA_W] > best_val) begin
                best_val = data[i*DATA_W +: DATA_W];
                best_idx = i[IDX_W-1:0];
            end
    end

    assign idx = best_idx;

endmodule
