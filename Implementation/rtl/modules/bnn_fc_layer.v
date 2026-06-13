`timescale 1ns / 1ps

// ============================================================
//  bnn_fc_layer — Chunk-Serial Binary Fully-Connected Layer
//
//  Thay vi tinh tat ca N_IN bits trong 1 cycle (ton nhieu LUT),
//  xu ly CHUNK_W bits/cycle, tich luy popcount trong accumulator FF.
//
//  Tai sao thiet ke nay can bang LUT va FF:
//    - LUT:  chi can XNOR + popcount cho CHUNK_W bits (nho gon)
//    - FF:   accumulator luu ket qua trung gian (dung FF hop ly)
//    - Timing: critical path ngan (chi CHUNK_W-bit popcount)
//
//  Latency: ceil(N_IN/CHUNK_W) + 1 cycles (1 cycle load + N chunks)
// ============================================================
module bnn_fc_layer #(
    parameter N_IN    = 784,
    parameter N_OUT   = 64,
    parameter CHUNK_W = 32,
    parameter SEED    = 32'hDEAD_BEEF,
    parameter RW      = $clog2(N_IN + 1)    // Accumulator width (in param list for port use)
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,      // Pulse 1 cycle de bat dau
    input  wire [N_IN-1:0]          data,       // Input (phai stable khi start)
    output wire [N_OUT*RW-1:0]      result,     // Popcount (valid khi done=1)
    output reg                      done        // Pulse 1 cycle khi hoan thanh
);

    // ---- Derived parameters ----
    localparam N_PAD    = ((N_IN + CHUNK_W - 1) / CHUNK_W) * CHUNK_W;
    localparam N_CHUNKS = N_PAD / CHUNK_W;
    localparam CW       = $clog2(CHUNK_W + 1);      // Chunk popcount width
    localparam CNT_W    = (N_CHUNKS == 1) ? 1 : $clog2(N_CHUNKS);

    // ---- Elaboration-time weight generation (xorshift32) ----
    // Function chay luc COMPILE → ket qua la constant
    function automatic [N_IN-1:0] gen_weight;
        input integer neuron_idx;
        reg [31:0] s;
        integer b;
    begin
        s = SEED ^ neuron_idx[31:0];
        for (b = 0; b < N_IN; b = b + 1) begin
            s = s ^ (s << 13);
            s = s ^ (s >> 17);
            s = s ^ (s <<  5);
            gen_weight[b] = s[0];
        end
    end
    endfunction

    // ---- Control FSM Registers ----
    reg               active;
    reg [CNT_W-1:0]   chunk_cnt;
    reg               done_internal;

    // ---- Input data register (Shift Register) ----
    // Load 1 lan khi start, sau do dich phai CHUNK_W bits moi chu ky
    // de tranh dung MUX tree dong
    reg [N_PAD-1:0] data_reg;

    // Pad input to N_PAD bits (zero-pad MSBs)
    wire [N_PAD-1:0] data_padded;
    generate
        if (N_PAD > N_IN)
            assign data_padded = {{(N_PAD - N_IN){1'b0}}, data};
        else
            assign data_padded = data;
    endgenerate

    always @(posedge clk) begin
        if (!rst_n)
            data_reg <= {N_PAD{1'b0}};
        else if (start)
            data_reg <= data_padded;
        else if (active)
            data_reg <= data_reg >> CHUNK_W;
    end

    // Chunk du lieu luon nam o LSBs cua data_reg (0 LUT delay!)
    wire [CHUNK_W-1:0] d_chunk = data_reg[CHUNK_W-1:0];

    // Pipeline stage register for data chunk (shared by all PEs to save resources and avoid warnings)
    reg [CHUNK_W-1:0] d_chunk_reg;
    always @(posedge clk) begin
        if (!rst_n)
            d_chunk_reg <= {CHUNK_W{1'b0}};
        else
            d_chunk_reg <= d_chunk;
    end

    // ---- Control FSM ----

    always @(posedge clk) begin
        if (!rst_n) begin
            active        <= 1'b0;
            chunk_cnt     <= {CNT_W{1'b0}};
            done_internal <= 1'b0;
        end else begin
            done_internal <= 1'b0;
            if (start) begin
                active    <= 1'b1;
                chunk_cnt <= {CNT_W{1'b0}};
            end else if (active) begin
                if (chunk_cnt == N_CHUNKS[CNT_W-1:0] - 1'b1) begin
                    active <= 1'b0;
                    done_internal <= 1'b1;
                end else begin
                    chunk_cnt <= chunk_cnt + 1'b1;
                end
            end
        end
    end

    // ---- Delay registers to match pipelining (1 cycle delay) ----
    reg active_d1;
    reg start_d1;

    always @(posedge clk) begin
        if (!rst_n) begin
            active_d1 <= 1'b0;
            start_d1  <= 1'b0;
            done      <= 1'b0;
        end else begin
            active_d1 <= active;
            start_d1  <= start;
            done      <= done_internal; // done tre 1 cycle theo accumulator
        end
    end

    // ---- Processing Elements (1 per output neuron) ----
    genvar g;
    generate
        for (g = 0; g < N_OUT; g = g + 1) begin : gen_pe

            // Padded weight constant:
            //   MSB padding = 1 → XNOR(data=0, weight=1) = 0
            //   → padding bits KHONG anh huong popcount
            localparam [N_IN-1:0]  W_RAW = gen_weight(g);
            wire [N_PAD-1:0] W_PAD;
            if (N_PAD > N_IN) begin : gen_w_pad
                assign W_PAD = {{(N_PAD - N_IN){1'b1}}, W_RAW};
            end else begin : gen_w_no_pad
                assign W_PAD = W_RAW;
            end

            // Weight chunk MUX (unrolled statically by the compiler)
            reg [CHUNK_W-1:0] w_chunk;
            integer c;
            always @(*) begin
                w_chunk = {CHUNK_W{1'b0}};
                for (c = 0; c < N_CHUNKS; c = c + 1) begin
                    if (chunk_cnt == c[CNT_W-1:0]) begin
                        w_chunk = W_PAD[c*CHUNK_W +: CHUNK_W];
                    end
                end
            end

            // Pipeline stage registers (breaking timing path in half)
            reg [CHUNK_W-1:0] w_chunk_reg;

            always @(posedge clk) begin
                if (!rst_n) begin
                    w_chunk_reg <= {CHUNK_W{1'b0}};
                end else begin
                    w_chunk_reg <= w_chunk;
                end
            end

            // XNOR + Popcount cho CHUNK_W bits using pipelined inputs
            wire [CW-1:0] chunk_pop;
            bnn_neuron #(.N(CHUNK_W)) u_pe (
                .data    (d_chunk_reg),
                .weight  (w_chunk_reg),
                .popcount(chunk_pop)
            );

            // Accumulator register (FF) — luu ket qua trung gian (delayed by 1 cycle)
            reg [RW-1:0] acc;
            always @(posedge clk) begin
                if (!rst_n || start_d1)
                    acc <= {RW{1'b0}};
                else if (active_d1)
                    acc <= acc + chunk_pop;
            end

            assign result[g*RW +: RW] = acc;
        end
    endgenerate

endmodule
