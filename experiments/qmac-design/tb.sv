`timescale 1ns/1ps
`include "vector_count.vh"
module tb;
    reg clk;
    // Testbench schedule only; NOT a measured achievable clock.
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    reg rst = 1;
    reg start = 0;
    reg inject_busy_requests = 0;
    reg [63:0] a = 0, w = 0;
    reg [31:0] acc = 0;
    wire [2:0] busy, done;
    wire [31:0] result [0:2];
    reg [191:0] vectors [0:`VECTOR_COUNT-1];
    reg [2:0] seen;
    reg [31:0] expected;
    integer i, j, elapsed, expected_latency, reset_delay;

    qmac8_unit #(.LANES(1)) u1(clk,rst,start | (inject_busy_requests & busy[0]),a,w,acc,busy[0],done[0],result[0]);
    qmac8_unit #(.LANES(2)) u2(clk,rst,start | (inject_busy_requests & busy[1]),a,w,acc,busy[1],done[1],result[1]);
    qmac8_unit #(.LANES(8)) u8(clk,rst,start | (inject_busy_requests & busy[2]),a,w,acc,busy[2],done[2],result[2]);

    task check_reset;
        begin
            @(negedge clk); rst = 1; start = 0;
            @(posedge clk); #1;
            if (busy !== 0 || done !== 0)
                $fatal(1,"Reset failed to cancel operation");
            for (j=0; j<3; j=j+1)
                if (result[j] !== 0) $fatal(1,"Reset result not cleared");
            @(negedge clk); rst = 0;
        end
    endtask

    initial begin
        $readmemh("vectors.mem",vectors);
        check_reset();
        // Cancel in-flight work at multiple phases, then test recovery below.
        for (reset_delay=1; reset_delay<=7; reset_delay=reset_delay+2) begin
            @(negedge clk); start=1; a=64'h8080808080808080;
            w=a; acc=32'h7fffffff;
            @(posedge clk); #1;
            @(negedge clk); start=0;
            repeat(reset_delay) begin @(posedge clk); #1; end
            check_reset();
            repeat(10) begin
                @(posedge clk); #1;
                if (busy !== 0 || done !== 0) $fatal(1,"Stale completion after reset");
            end
        end
        for (i=0; i<`VECTOR_COUNT; i=i+1) begin
            @(negedge clk);
            {a,w,acc,expected}=vectors[i];
            start=1;
            @(posedge clk); #1; // Acceptance edge: latency starts here.
            if (busy !== 3'b111 || done !== 0) $fatal(1,"Request not accepted");
            @(negedge clk);
            start=0;
            inject_busy_requests=1;
            // Poison external inputs; all designs must use their latched request.
            a=~a; w=~w; acc=~acc;
            seen=0;
            for (elapsed=1; elapsed<=9; elapsed=elapsed+1) begin
                @(posedge clk); #1;
                for (j=0; j<3; j=j+1) begin
                    if (j==0) expected_latency=8;
                    else if (j==1) expected_latency=4;
                    else expected_latency=1;
                    if (elapsed==expected_latency) begin
                        if (done[j] !== 1 || busy[j] !== 0)
                            $fatal(1,"Wrong completion timing case=%0d unit=%0d elapsed=%0d",i,j,elapsed);
                        if (result[j] !== expected)
                            $fatal(1,"Mismatch case=%0d unit=%0d got=%h expected=%h",i,j,result[j],expected);
                        seen[j]=1;
                    end else begin
                        if (done[j] !== 0) $fatal(1,"Unexpected done pulse");
                        if (elapsed<expected_latency && busy[j] !== 1)
                            $fatal(1,"Busy dropped early");
                        if (elapsed>expected_latency && busy[j] !== 0)
                            $fatal(1,"Unexpected new request");
                    end
                end
            end
            if (seen !== 3'b111) $fatal(1,"Missing completion");
        end
        $display("PASS: %0d vectors x 3 designs; latency LANES=1:8, LANES=2:4, LANES=8:1 cycles.",`VECTOR_COUNT);
        $display("PASS: reset cancellation/recovery, latched inputs, ignored busy requests, single done pulse.");
        $finish;
    end
    initial begin
        #2000000;
        $fatal(1,"Timeout");
    end
endmodule
