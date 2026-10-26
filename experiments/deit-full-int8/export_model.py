#!/usr/bin/env python3
"""Full synthetic-input DeiT-Tiny W8A8 execution fixture and integer reference.

This is a project-specific quantization profile, NOT calibrated ImageNet accuracy
or generic LiteRT builtin operator compatibility. Every Transformer block runs.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import flatbuffers
import numpy as np
import timm
import torch
import tflite as t

ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'build'
D,T,H,HEADS=192,197,768,3

def away(a):
    return np.copysign(np.floor(np.abs(a)+.5),a)
def scale(a):
    return np.float32(max(float(np.max(np.abs(a))),1e-9)/127)
def quant(a,s):
    return np.clip(away(np.asarray(a,dtype=np.float64)/float(s)),-127,127).astype(np.int8)
def qm(r):
    f,s=math.frexp(float(r)); m=math.floor(f*(1<<31)+.5)
    if m==1<<31: m//=2;s+=1
    if s< -31: return 0,0
    assert -31<=s<=30
    return m,s
def rq(a,m,s,clamp=True):
    bits=31-np.asarray(s,dtype=np.int64)
    v=np.asarray(a,dtype=np.int64)*np.asarray(m,dtype=np.int64)+(np.int64(1)<<(bits-1))
    q=v>>bits
    assert np.max(np.abs(q))<(1<<31)
    return np.clip(q,-128,127).astype(np.int8) if clamp else q
def fnv(a):
    h=2166136261
    for v in np.asarray(a,dtype=np.int8).tobytes(): h=((h^v)*16777619)&0xffffffff
    return h
def bits(s): return struct.unpack('<I',struct.pack('<f',float(s)))[0]

class Blob:
    def __init__(self,kind,sx,sy):
        self.meta=[0]*64;self.meta[0]=kind;self.meta[3]=bits(sx);self.meta[4]=bits(sy)
        self.data=bytearray(256)
    def add(self,a):
        self.data.extend(b'\0'*((-len(self.data))%16)); off=len(self.data)
        self.data.extend(np.asarray(a).tobytes()); return off
    def params(self,index,*values):
        self.meta[index:index+len(values)]=[int(v)&0xffffffff for v in values]
    def finish(self,y):
        self.meta[2]=fnv(y);self.data[:256]=struct.pack('<64I',*self.meta)
        return bytes(self.data)

def dense(blob,index,module,sx,sy):
    w=module.weight.detach().numpy().astype(np.float64).reshape(module.weight.shape[0],-1)
    sw=np.maximum(np.max(np.abs(w),axis=1)/127,1e-12).astype(np.float32)
    wq=np.clip(away(w/sw[:,None]),-127,127).astype(np.int8)
    bias=module.bias.detach().numpy().astype(np.float64) if module.bias is not None else np.zeros(len(w))
    bq=away(bias/(float(sx)*sw.astype(np.float64))).astype(np.int64)
    assert np.max(np.abs(bq))<(1<<31)
    bound=128*np.abs(wq.astype(np.int64)).sum(axis=1)+np.abs(bq)
    assert np.max(bound)<(1<<31)
    pairs=[qm(float(sx)*float(s)/float(sy)) for s in sw]
    m=np.array([p[0] for p in pairs],dtype='<i4');s=np.array([p[1] for p in pairs],dtype='<i4')
    bq=bq.astype('<i4')
    blob.params(index,blob.add(wq),blob.add(bq),blob.add(m),blob.add(s))
    return (wq,bq,m,s)
def linear(x,p,lut=None):
    w,b,m,s=p
    acc=x.astype(np.int64)@w.astype(np.int64).T+b
    assert np.max(np.abs(acc))<(1<<31)
    y=rq(acc,m,s)
    return lut[y.astype(np.int16)+128] if lut is not None else y

def norm_params(blob,index,module,sx,sy):
    g=away(module.weight.detach().numpy().astype(np.float64)/float(sy)*(1<<16)).astype(np.int64)
    b=away(module.bias.detach().numpy().astype(np.float64)/float(sy)).astype(np.int64)
    eps=max(1,int(math.floor(float(module.eps)*D*D/(float(sx)**2)+.5)))
    assert eps<(1<<32) and np.max(np.abs(g))<(1<<31) and np.max(np.abs(b))<(1<<31)
    g=g.astype('<i4');b=b.astype('<i4')
    blob.params(index,blob.add(g),blob.add(b),eps)
    return g,b,eps
def layernorm(x,p):
    g,b,eps=p;z=x.astype(np.int64);s=z.sum(axis=1);ss=(z*z).sum(axis=1)
    var=D*ss-s*s+eps
    denom=np.array([math.isqrt(int(v)) for v in var],dtype=np.int64)
    numerator=(D*z-s[:,None])*g[None,:]
    divisor=denom[:,None]*(1<<16)
    q=(numerator+divisor//2)//divisor+b
    return np.clip(q,-128,127).astype(np.int8)
def residual(a,b,sx,sp,sy,blob,index):
    m0,s0=qm(float(sx)/float(sy));m1,s1=qm(float(sp)/float(sy))
    blob.params(index,m0,s0,m1,s1)
    return np.clip(rq(a,m0,s0,False)+rq(b,m1,s1,False),-128,127).astype(np.int8)

def lut_gelu(sx,sy):
    table=[]
    for q in range(-128,128):
        x=q*float(sx);real=.5*x*math.erfc(-x/math.sqrt(2))
        table.append(max(-128,min(127,int(away(real/float(sy))))))
    table=np.array(table,dtype=np.int8)
    real=torch.nn.functional.gelu(torch.arange(-128,128,dtype=torch.float64)*float(sx),approximate='none')/float(sy)
    independent=(torch.sign(real)*torch.floor(torch.abs(real)+.5)).clamp(-128,127).to(torch.int8).numpy()
    np.testing.assert_array_equal(table,independent)
    return table

def attention(qkv,sq,ss,sc,blob):
    m,s=qm(float(sq)*float(sq)/8/float(ss));blob.params(24,m,s)
    exp=np.array([math.floor(math.exp(-i*float(ss))*(1<<24)+.5) for i in range(256)],dtype='<u4')
    blob.params(26,blob.add(exp))
    vm,vs=qm(float(sq)/255/float(sc));blob.params(27,vm,vs)
    heads=[];parts=qkv.reshape(T,3,HEADS,64)
    for h in range(HEADS):
        q,k,v=[parts[:,i,h,:].astype(np.int64) for i in range(3)]
        scores=rq(q@k.T,m,s)
        diff=scores.max(axis=1).astype(np.int16)[:,None]-scores.astype(np.int16)
        ex=exp[diff].astype(np.int64);sums=ex.sum(axis=1)[:,None]
        # UINT8 probabilities at scale 1/255. C encodes these as signed q=p-128
        # for qmac8, with an explicit zero-point correction in the AV bias.
        probs=(ex*255+sums//2)//sums
        context=rq(probs@v,vm,vs)
        heads.append(context)
    return np.concatenate(heads,axis=1)

def make_flatbuffer(blobs,scales,sinput):
    b=flatbuffers.Builder(8*1024*1024)
    def vec(vals,kind='i'):
        b.StartVector(4,len(vals),4)
        for v in reversed(vals):
            (b.PrependUOffsetTRelative if kind=='o' else b.PrependInt32)(v)
        return b.EndVector()
    def qparams(s):
        b.StartVector(4,1,4);b.PrependFloat32(float(s));sv=b.EndVector()
        b.StartVector(8,1,8);b.PrependInt64(0);zv=b.EndVector()
        t.QuantizationParametersStart(b);t.QuantizationParametersAddScale(b,sv);t.QuantizationParametersAddZeroPoint(b,zv)
        return t.QuantizationParametersEnd(b)
    buffers=[]
    for data in [b'']+blobs:
        if data: b.Prep(16,len(data));raw=b.CreateByteVector(data)
        t.BufferStart(b)
        if data: t.BufferAddData(b,raw)
        buffers.append(t.BufferEnd(b))
    tensors=[]
    # Tensors 0..14 are model input and the 14 stage outputs. 15..28 are blobs.
    for i in range(15):
        shape=[1,3,224,224] if i==0 else ([1,1000] if i==14 else [1,T,D])
        sv=vec(shape);quant=qparams(sinput if i==0 else scales[i-1])
        t.TensorStart(b);t.TensorAddShape(b,sv);t.TensorAddType(b,t.TensorType.INT8);t.TensorAddBuffer(b,0)
        t.TensorAddQuantization(b,quant);t.TensorAddHasRank(b,True);tensors.append(t.TensorEnd(b))
    for i,data in enumerate(blobs):
        sv=vec([len(data)]);t.TensorStart(b);t.TensorAddShape(b,sv);t.TensorAddType(b,t.TensorType.UINT8)
        t.TensorAddBuffer(b,i+1);t.TensorAddHasRank(b,True);tensors.append(t.TensorEnd(b))
    operators=[]
    for i in range(14):
        ins=vec([i,15+i]);outs=vec([i+1])
        t.OperatorStart(b);t.OperatorAddOpcodeIndex(b,0 if i==0 else (2 if i==13 else 1))
        t.OperatorAddInputs(b,ins);t.OperatorAddOutputs(b,outs);operators.append(t.OperatorEnd(b))
    tv,iv,ov,ops=vec(tensors,'o'),vec([0]),vec([14]),vec(operators,'o')
    t.SubGraphStart(b);t.SubGraphAddTensors(b,tv);t.SubGraphAddInputs(b,iv);t.SubGraphAddOutputs(b,ov);t.SubGraphAddOperators(b,ops)
    graph=t.SubGraphEnd(b)
    codes=[]
    for name in ['DEIT_EMBED_INT8','DEIT_BLOCK_INT8','DEIT_HEAD_INT8']:
        off=b.CreateString(name);t.OperatorCodeStart(b);t.OperatorCodeAddBuiltinCode(b,t.BuiltinOperator.CUSTOM)
        t.OperatorCodeAddDeprecatedBuiltinCode(b,t.BuiltinOperator.CUSTOM);t.OperatorCodeAddCustomCode(b,off)
        t.OperatorCodeAddVersion(b,1);codes.append(t.OperatorCodeEnd(b))
    graphs,codesv,bufs=vec([graph],'o'),vec(codes,'o'),vec(buffers,'o')
    desc=b.CreateString('Full DeiT-Tiny: project W8A8 matrices, integer LN/Softmax/GELU, synthetic verification; NOT calibrated accuracy')
    t.ModelStart(b);t.ModelAddVersion(b,3);t.ModelAddOperatorCodes(b,codesv);t.ModelAddSubgraphs(b,graphs)
    t.ModelAddBuffers(b,bufs);t.ModelAddDescription(b,desc);model=t.ModelEnd(b);b.Finish(model,file_identifier=b'TFL3')
    return bytes(b.Output())

def main():
    BUILD.mkdir(exist_ok=True)
    source=ROOT.parent/'deit-tiny-inspect'
    manifest=json.loads((source/'results/model-manifest.json').read_text())
    ckpt=ROOT.parent/'deit-tiny-inspect'/manifest['checkpoint_file']
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=timm.create_model(manifest['model'],pretrained=False)
    model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
    fixture=torch.load(source/'results/float-fixtures.pt',map_location='cpu',weights_only=True)
    image=fixture['input']
    for block in model.blocks: block.attn.fused_attn=False
    def array(x): return x.detach().numpy().reshape(-1,x.shape[-1]).astype(np.float64)
    records=[]
    with torch.inference_mode():
        patches=model.patch_embed(image);tok=torch.cat([model.cls_token.expand(1,-1,-1),patches],dim=1)+model.pos_embed
        initial=array(tok);patchfloat=array(patches)
        for block in model.blocks:
            n1=block.norm1(tok);qkv=block.attn.qkv(n1)
            q,k,v=qkv.reshape(1,T,3,HEADS,64).permute(2,0,3,1,4).unbind(0)
            scores=(q@k.transpose(-2,-1))*.125
            ctx=(scores.softmax(-1)@v).transpose(1,2).reshape(1,T,D)
            proj=block.attn.proj(ctx);r1=tok+proj;n2=block.norm2(r1)
            f1=block.mlp.fc1(n2);g=block.mlp.act(f1);f2=block.mlp.fc2(g);after=r1+f2
            records.append(dict(n1=array(n1),qkv=array(qkv),scores=scores.numpy(),ctx=array(ctx),proj=array(proj),
                r1=array(r1),n2=array(n2),f1=array(f1),g=array(g),f2=array(f2),after=array(after)))
            tok=after
        finalnorm=model.norm(tok);logits=model.head(finalnorm[:,0])
        torch.testing.assert_close(logits,model(image),rtol=1e-4,atol=1e-5)
    sx=scale(image.numpy());s0=scale(initial);sp=scale(patchfloat)
    xq=quant(image.numpy(),sx).reshape(3,224,224)
    embed=Blob(0,sx,s0);dp=dense(embed,20,model.patch_embed.proj,sx,sp)
    im2col=np.stack([xq[:,py*16:(py+1)*16,px*16:(px+1)*16].reshape(-1) for py in range(14) for px in range(14)])
    patchq=linear(im2col,dp)
    pos=away(model.pos_embed.detach().numpy().reshape(T,D).astype(np.float64)/float(s0)).astype('<i4')
    cls=away(model.cls_token.detach().numpy().reshape(D).astype(np.float64)/float(s0)).astype('<i4')
    pm,ps=qm(float(sp)/float(s0));embed.params(24,embed.add(pos),embed.add(cls),pm,ps)
    tokenq=np.empty((T,D),dtype=np.int8);tokenq[0]=np.clip(cls+pos[0],-128,127)
    tokenq[1:]=np.clip(rq(patchq,pm,ps,False)+pos[1:],-128,127)
    blobs=[embed.finish(tokenq)];scales=[s0];stagehashes=[fnv(tokenq)];current=s0
    for i,(block,r) in enumerate(zip(model.blocks,records)):
        so={name:scale(value) for name,value in r.items()}
        blob=Blob(1,current,so['after']);blob.meta[1]=i
        n1=layernorm(tokenq,norm_params(blob,16,block.norm1,current,so['n1']))
        qkv=linear(n1,dense(blob,20,block.attn.qkv,so['n1'],so['qkv']))
        ctx=attention(qkv,so['qkv'],so['scores'],so['ctx'],blob)
        proj=linear(ctx,dense(blob,32,block.attn.proj,so['ctx'],so['proj']))
        r1=residual(tokenq,proj,current,so['proj'],so['r1'],blob,36)
        n2=layernorm(r1,norm_params(blob,40,block.norm2,so['r1'],so['n2']))
        lut=lut_gelu(so['f1'],so['g']);blob.params(48,blob.add(lut))
        hidden=linear(n2,dense(blob,44,block.mlp.fc1,so['n2'],so['f1']),lut)
        f2=linear(hidden,dense(blob,52,block.mlp.fc2,so['g'],so['f2']))
        tokenq=residual(r1,f2,so['r1'],so['f2'],so['after'],blob,56)
        blobs.append(blob.finish(tokenq));scales.append(so['after']);stagehashes.append(fnv(tokenq));current=so['after']
        print('Integer reference block',i+1,'hash',hex(stagehashes[-1]),flush=True)
    sn=scale(array(finalnorm));sl=scale(array(logits))
    head=Blob(2,current,sl);normq=layernorm(tokenq,norm_params(head,16,model.norm,current,sn))
    result=linear(normq[:1],dense(head,20,model.head,sn,sl))
    blobs.append(head.finish(result));scales.append(sl);stagehashes.append(fnv(result))
    raw=make_flatbuffer(blobs,scales,sx);(BUILD/'model.tflite').write_bytes(raw)
    (BUILD/'input.bin').write_bytes(xq.tobytes());(BUILD/'expected.bin').write_bytes(result.tobytes())
    np.savez_compressed(BUILD/'reference.npz',input=xq,logits=result,stage_hashes=stagehashes,float_logits=logits.numpy(),output_scale=sl)
    h=['#include <stdint.h>',f'#define FULL_MODEL_BYTES {len(raw)}',f'#define FULL_OUTPUT_SCALE {float(sl):.17g}',
       'static const int8_t expected_logits[1000]={'+','.join(str(int(v)) for v in result.reshape(-1))+'};']
    (BUILD/'reference.h').write_text('\n'.join(h)+'\n')
    asm=['.section .rodata.full_model,"a",@progbits','.balign 16','.global full_model_data','full_model_data:',
         '.incbin "'+str(BUILD/'model.tflite')+'"','.balign 16','.global full_input_data','full_input_data:',
         '.incbin "'+str(BUILD/'input.bin')+'"','.section .note.GNU-stack,"",@progbits']
    (BUILD/'model.S').write_text('\n'.join(asm)+'\n')
    report=dict(model=manifest['model'],blocks=12,tokens=T,embedding_dim=D,hidden_dim=H,heads=HEADS,
        graph_nodes=14,checkpoint_sha256=manifest['checkpoint_sha256'],model_sha256=hashlib.sha256(raw).hexdigest(),
        input_sha256=hashlib.sha256(xq.tobytes()).hexdigest(),model_bytes=len(raw),quantized_top1=int(np.argmax(result)),
        float_top1=int(torch.argmax(logits)),stage_hashes=stagehashes,output_scale=float(sl),
        profile='Project W8A8 matrices; INT32 bias; fixed-point Q16 LayerNorm affine; integer isqrt variance; Q24 exp-LUT Softmax with UINT8 probabilities; Plan-A GELU LUT; static per-tensor activations/per-channel matrix weights',
        limitations=['Synthetic input and scale selection, not representative calibration or dataset accuracy',
                    'Project integer LayerNorm/Softmax approximations, not LiteRT builtin bit-exact compatibility',
                    'Custom TFLM graph operators contain embedding, complete Transformer blocks and final head',
                    'All learned matrix weights included; norm affine/position constants use wider integer metadata'],
        full_path=['Patch embedding','CLS + positional addition','12 complete attention/MLP/residual blocks','Final LayerNorm','CLS classifier 1000 logits'])
    (BUILD/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
