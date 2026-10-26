#!/usr/bin/env python3
"""Export one model-derived full fc1 operator, not a converted full DeiT model."""
import hashlib
import json
import argparse
import math
import struct
from pathlib import Path

import flatbuffers
import numpy as np
import tflite as t

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'


def c_array(name, dtype, data):
    values = np.asarray(data).reshape(-1)
    lines = [f'alignas(16) static const {dtype} {name}[] = {{']
    for i in range(0, len(values), 24):
        lines.append(','.join(str(int(v)) for v in values[i:i+24]) + ',')
    return '\n'.join(lines + ['};'])


def export(rows, channels, source, gelu=False, fused=False):
    from math import frexp, floor
    dest = BUILD / (f'{rows}x{channels}' + ('-gelu' if gelu else '') + ('-fused' if fused else ''))
    dest.mkdir(parents=True, exist_ok=True)
    x = source['input_q'][:rows].copy()
    w = source['weight_q'][:channels].copy()
    bias = source['bias_q'][:channels].astype('<i4')
    # The FlatBuffer stores FLOAT32 scales. Recompute the reference from those
    # exact serialized values rather than assuming float64 export is bit-exact.
    sx, sy = np.float32(source['input_scale']), np.float32(source['output_scale'])
    sw = source['weight_scales'][:channels].astype(np.float32)
    bound = 128*np.abs(w.astype(np.int64)).sum(axis=1) + np.abs(bias.astype(np.int64))
    assert int(bound.max()) < (1 << 31)
    acc = x.astype(np.int64) @ w.astype(np.int64).T + bias
    multipliers, shifts = [], []
    for scale in float(sx) * sw.astype(np.float64) / float(sy):
        fraction, shift = frexp(float(scale))
        multiplier = floor(fraction * (1 << 31) + .5)
        if multiplier == 1 << 31:
            multiplier //= 2
            shift += 1
        assert -31 <= shift <= 30
        multipliers.append(multiplier)
        shifts.append(shift)
    y = np.empty_like(acc, dtype=np.int8)
    for row in range(rows):
        for channel in range(channels):
            bits = 31 - shifts[channel]
            value = (int(acc[row, channel])*multipliers[channel] + (1 << (bits-1))) // (1 << bits)
            y[row, channel] = min(127, max(-128, value))
    # Independent arbitrary-precision checks at fixed boundaries/sampled points.
    for row, channel in {(0, 0), (rows-1, channels-1), (rows//2, channels//2)}:
        assert int(acc[row, channel]) == int(bias[channel]) + sum(
            int(a)*int(b) for a, b in zip(x[row], w[channel]))

    fc_y = y.copy()
    if gelu:
        import torch
        # Plan A: exact GELU on dequantized INT8 fc1 outputs. Initially retain
        # the fc1 scale as the output scale; this is an operator fixture choice,
        # not dataset calibration. Both scales are explicitly stored in model.
        gelu_scale = sy
        def encode(value):
            scaled = value/float(gelu_scale)
            rounded = math.copysign(math.floor(abs(scaled)+0.5), scaled)
            return max(-128, min(127, int(rounded)))
        lut = np.array([encode(0.5*(q*float(sy))*math.erfc(-q*float(sy)/math.sqrt(2)))
                        for q in range(-128,128)],dtype=np.int8)
        # Independent reference uses PyTorch exact GELU, not a LUT lookup.
        def torch_reference(q):
            x = torch.from_numpy(np.asarray(q).astype(np.float64))*float(sy)
            real = torch.nn.functional.gelu(x, approximate='none') / float(gelu_scale)
            return (torch.sign(real)*torch.floor(torch.abs(real)+0.5)).clamp(-128,127).to(torch.int8).numpy()
        expected_lut = torch_reference(np.arange(-128,128))
        np.testing.assert_array_equal(lut,expected_lut)
        y = torch_reference(fc_y)
        np.testing.assert_array_equal(y, lut[fc_y.astype(np.int16)+128])
        # Fixed little-endian custom options: magic, scales, zero points, LUT.
        gelu_options = struct.pack('<4sffii',b'GLU1',float(sy),float(gelu_scale),0,0)+lut.tobytes()

    b = flatbuffers.Builder(256*1024)
    def vector(values, kind):
        funcs = {'i': (b.PrependInt32, 4, 4), 'l': (b.PrependInt64, 8, 8),
                 'f': (b.PrependFloat32, 4, 4), 'o': (b.PrependUOffsetTRelative, 4, 4)}
        prepend, size, alignment = funcs[kind]
        b.StartVector(size, len(values), alignment)
        for v in reversed(values):
            prepend(v)
        return b.EndVector()
    def quant(scales, zeros, axis=0):
        sv, zv = vector(scales, 'f'), vector(zeros, 'l')
        t.QuantizationParametersStart(b)
        t.QuantizationParametersAddScale(b, sv)
        t.QuantizationParametersAddZeroPoint(b, zv)
        t.QuantizationParametersAddQuantizedDimension(b, axis)
        return t.QuantizationParametersEnd(b)
    buffers = []
    for raw in [b'', w.tobytes(), bias.tobytes()]:
        if raw:
            b.Prep(16, len(raw))
        data = b.CreateByteVector(raw) if raw else None
        t.BufferStart(b)
        if data is not None:
            t.BufferAddData(b, data)
        buffers.append(t.BufferEnd(b))
    tensors = []
    specs = [('input', [rows, 192], t.TensorType.INT8, 0, [float(sx)], [0]),
             ('weights', [channels, 192], t.TensorType.INT8, 1, sw.tolist(), [0]*channels),
             ('bias', [channels], t.TensorType.INT32, 2, (sx*sw).tolist(), [0]*channels),
             ('output', [rows, channels], t.TensorType.INT8, 0, [float(sy)], [0])]
    if gelu and not fused:
        specs.append(('gelu_output',[rows,channels],t.TensorType.INT8,0,[float(gelu_scale)],[0]))
    for name, shape, dtype, buffer, scales, zeros in specs:
        nameoff, shapeoff = b.CreateString(name), vector(shape, 'i')
        q = quant(scales, zeros)
        t.TensorStart(b)
        t.TensorAddShape(b, shapeoff)
        t.TensorAddType(b, dtype)
        t.TensorAddBuffer(b, buffer)
        t.TensorAddName(b, nameoff)
        t.TensorAddQuantization(b, q)
        t.TensorAddHasRank(b, True)
        tensors.append(t.TensorEnd(b))
    t.FullyConnectedOptionsStart(b)
    t.FullyConnectedOptionsAddFusedActivationFunction(b, t.ActivationFunctionType.NONE)
    options = t.FullyConnectedOptionsEnd(b)
    ins, outs = vector([0, 1, 2], 'i'), vector([3], 'i')
    fused_options=b.CreateByteVector(gelu_options) if fused else None
    t.OperatorStart(b)
    t.OperatorAddOpcodeIndex(b, 0)
    t.OperatorAddInputs(b, ins)
    t.OperatorAddOutputs(b, outs)
    if fused:
        t.OperatorAddCustomOptions(b,fused_options)
    else:
        t.OperatorAddBuiltinOptionsType(b, t.BuiltinOptions.FullyConnectedOptions)
        t.OperatorAddBuiltinOptions(b, options)
    op = t.OperatorEnd(b)
    operators=[op]
    if gelu and not fused:
        ins,outs,custom=vector([3],'i'),vector([4],'i'),b.CreateByteVector(gelu_options)
        t.OperatorStart(b)
        t.OperatorAddOpcodeIndex(b,1)
        t.OperatorAddInputs(b,ins)
        t.OperatorAddOutputs(b,outs)
        t.OperatorAddCustomOptions(b,custom)
        t.OperatorAddCustomOptionsFormat(b,t.CustomOptionsFormat.FLEXBUFFERS)
        operators.append(t.OperatorEnd(b))
    tv, iv, ov, ops = vector(tensors, 'o'), vector([0], 'i'), vector([4] if gelu and not fused else [3], 'i'), vector(operators, 'o')
    t.SubGraphStart(b)
    t.SubGraphAddTensors(b, tv)
    t.SubGraphAddInputs(b, iv)
    t.SubGraphAddOutputs(b, ov)
    t.SubGraphAddOperators(b, ops)
    graph = t.SubGraphEnd(b)
    fused_name=b.CreateString('FC_GELU_LUT') if fused else None
    t.OperatorCodeStart(b)
    t.OperatorCodeAddBuiltinCode(b, t.BuiltinOperator.CUSTOM if fused else t.BuiltinOperator.FULLY_CONNECTED)
    t.OperatorCodeAddDeprecatedBuiltinCode(b, t.BuiltinOperator.CUSTOM if fused else t.BuiltinOperator.FULLY_CONNECTED)
    if fused: t.OperatorCodeAddCustomCode(b,fused_name)
    t.OperatorCodeAddVersion(b, 1 if fused else 4)
    opcode = t.OperatorCodeEnd(b)
    opcodes=[opcode]
    if gelu and not fused:
        name=b.CreateString('GELU_LUT')
        t.OperatorCodeStart(b)
        t.OperatorCodeAddBuiltinCode(b,t.BuiltinOperator.CUSTOM)
        t.OperatorCodeAddDeprecatedBuiltinCode(b,t.BuiltinOperator.CUSTOM)
        t.OperatorCodeAddCustomCode(b,name)
        t.OperatorCodeAddVersion(b,1)
        opcodes.append(t.OperatorCodeEnd(b))
    graphs, codes, bufs = vector([graph], 'o'), vector(opcodes, 'o'), vector(buffers, 'o')
    description = b.CreateString('DeiT-Tiny fc1 operator verification; synthetic activations; no accuracy claim')
    t.ModelStart(b)
    t.ModelAddVersion(b, 3)
    t.ModelAddOperatorCodes(b, codes)
    t.ModelAddSubgraphs(b, graphs)
    t.ModelAddBuffers(b, bufs)
    t.ModelAddDescription(b, description)
    model = t.ModelEnd(b)
    b.Finish(model, file_identifier=b'TFL3')
    raw = bytes(b.Output())
    (dest/'model.tflite').write_bytes(raw)
    parsed = t.Model.GetRootAsModel(raw, 0)
    assert parsed.Subgraphs(0).Tensors(0).Shape(0) == rows
    header = ['#include <stdint.h>', f'#define TEST_M {rows}\n#define TEST_K 192\n#define TEST_N {channels}',
              c_array('model_data', 'uint8_t', np.frombuffer(raw, dtype=np.uint8)),
              c_array('input_data', 'int8_t', x), c_array('expected_y', 'int8_t', y),
              c_array('expected_multipliers', 'int32_t', multipliers),
              c_array('expected_shifts', 'int32_t', shifts)]
    if gelu:
        header += [c_array('expected_fc_y','int8_t',fc_y),
                   c_array('expected_gelu_lut','int8_t',expected_lut)]
    (dest/'model_data.h').write_text('\n'.join(header)+'\n')
    report = dict(M=rows, K=192, N=channels, outputs=rows*channels, macs=rows*192*channels,
                  conservative_acc_bound_all_int8_inputs=int(bound.max()),
                  model_bytes=len(raw), model_sha256=hashlib.sha256(raw).hexdigest(),
                  scope='One FullyConnected operator, actual pretrained weights and synthetic activations',
                  rounding='TFLITE_SINGLE_ROUNDING=1; scales serialized as float32',
                  changes_from_prior_float64_scale_reference=int(np.count_nonzero(fc_y != source['output_q'][:rows,:channels])),
                  input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                  output_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    if gelu:
        report.update(scope='FullyConnected -> exact GELU LUT; actual weights, synthetic activations',
            graph_operators=['FC_GELU_LUT'] if fused else ['FULLY_CONNECTED','GELU_LUT'],gelu_input_scale=float(sy),
            gelu_output_scale=float(gelu_scale),gelu_input_zero_point=0,gelu_output_zero_point=0,
            gelu_lut_bytes=256,gelu_options_bytes=len(gelu_options),
            gelu_formula='0.5*x*erfc(-x/sqrt(2)), approximate=none',
            gelu_rounding='nearest, ties away from zero, then INT8 clamp',
            gelu_lut_sha256=hashlib.sha256(lut.tobytes()).hexdigest(),
            gelu_reference='Independent PyTorch float64 exact GELU + quantization, all 256 inputs and every full output',
            fc_output_sha256=hashlib.sha256(fc_y.tobytes()).hexdigest())
    (dest/'reference.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gelu',action='store_true')
    parser.add_argument('--fused',action='store_true')
    args=parser.parse_args()
    if args.fused: args.gelu=True
    source = np.load(ROOT.parent/'deit-linear-int8/build/operator-reference.npz')
    for dims in [(2, 8), (197, 768)]:
        export(*dims, source, gelu=args.gelu, fused=args.fused)
