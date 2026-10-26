`timescale 1ns/1ps
// Standalone design experiment, not integrated into Spike or CV-Wally.
// Experimental semantics: signed INT8 lanes, lane 0 = bits[7:0],
// result = low32(acc_in + sum(a[i] * w[i])). No saturation/zero-point correction.
// Accept start when !busy. Inputs are latched. done pulses on result completion.
// Synchronous reset cancels an in-flight operation. No request overlap.
module qmac8_unit #(
    parameter integer LANES = 8
) (
    input wire clk,
    input wire rst,
    input wire start,
    input wire [63:0] a,
    input wire [63:0] w,
    input wire [31:0] acc_in,
    output reg busy,
    output reg done,
    output reg [31:0] result
);
    localparam integer CHUNKS = 8 / LANES;
    localparam [2:0] LAST_STEP = 3'(CHUNKS-1);
    reg [63:0] a_hold, w_hold;
    reg [31:0] accumulator;
    reg [2:0] step;
    wire signed [15:0] product [0:LANES-1];
    wire signed [18:0] chunk_dot;
    wire [31:0] accumulated;

    genvar k;
    generate
        for (k = 0; k < LANES; k = k + 1) begin: multipliers
            if (LANES == 8) begin: fixed_lanes
                assign product[k] = $signed(a_hold[k*8 +: 8]) *
                                    $signed(w_hold[k*8 +: 8]);
            end else begin: selected_lanes
                assign product[k] = $signed(a_hold[(step*LANES+k)*8 +: 8]) *
                                    $signed(w_hold[(step*LANES+k)*8 +: 8]);
            end
        end
        if (LANES == 1) begin: serial_sum
            assign chunk_dot = {{3{product[0][15]}}, product[0]};
        end else if (LANES == 2) begin: pair_sum
            assign chunk_dot = $signed({{3{product[0][15]}}, product[0]}) +
                               $signed({{3{product[1][15]}}, product[1]});
        end else if (LANES == 8) begin: balanced_tree
            // Grow by one bit per level: 16 -> 17 -> 18 -> 19.
            wire signed [16:0] level1 [0:3];
            wire signed [17:0] level2 [0:1];
            for (k = 0; k < 4; k = k + 1) begin: first_level
                assign level1[k] = $signed({product[2*k][15], product[2*k]}) +
                                   $signed({product[2*k+1][15], product[2*k+1]});
            end
            for (k = 0; k < 2; k = k + 1) begin: second_level
                assign level2[k] = $signed({level1[2*k][16], level1[2*k]}) +
                                   $signed({level1[2*k+1][16], level1[2*k+1]});
            end
            assign chunk_dot = $signed({level2[0][17], level2[0]}) +
                               $signed({level2[1][17], level2[1]});
        end
    endgenerate

    assign accumulated = accumulator + {{13{chunk_dot[18]}}, chunk_dot};

    always @(posedge clk) begin
        if (rst) begin
            a_hold <= 0;
            w_hold <= 0;
            accumulator <= 0;
            step <= 0;
            busy <= 0;
            done <= 0;
            result <= 0;
        end else begin
            done <= 0;
            if (!busy) begin
                if (start) begin
                    a_hold <= a;
                    w_hold <= w;
                    accumulator <= acc_in;
                    step <= 0;
                    busy <= 1;
                end
            end else begin
                accumulator <= accumulated;
                if (step == LAST_STEP) begin
                    result <= accumulated;
                    busy <= 0;
                    done <= 1;
                end else begin
                    step <= step + 1'b1;
                end
            end
        end
    end
endmodule
