#!/usr/bin/env python3
# Copied from the research phase (scratchpad qwen3-tools). usage: replace_weights.py --input workloads/qwen3_1p7b/trace
#   --output <dir> --hbf <regex> [--first N --last M]  (kernel positions of the input kernelslist.g)
"""Re-place Qwen3 weight tensors in the bundled (already HBF-remapped) trace.

The bundled replay/manifest.json lists every parameter tensor with its original GPU address
('address') and its HBF-window address ('target'), so the remap is invertible.
Policy: tensors whose name matches --hbf REGEX are packed (4 KiB aligned, in original-address order)
from --base upward; all other tensors go back to their ORIGINAL addresses (ordinary GPU memory).
--hbf '^$' therefore restores the un-remapped (all-GPU-memory) addresses.
Non-weight addresses (KV cache, activations, scratch) are never touched.
"""
import argparse, bisect, json, lzma, re
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--hbf',default='.*',help='regex of tensor names placed in the HBF window'); ap.add_argument('--base',type=int,default=274877906944)
    ap.add_argument('--first',type=int,default=0); ap.add_argument('--last',type=int,default=10**9); a=ap.parse_args()
    man=json.loads((a.input/'manifest.json').read_text()); W=sorted(man['weights'],key=lambda w:w['target'])
    pat=re.compile(a.hbf); cur=a.base
    for w in sorted(W,key=lambda w:w['address']):
        if pat.search(w['name']): cur=(cur+4095)//4096*4096; w['dest']=cur; cur+=w['bytes']
        else: w['dest']=w['address']
    T=[w['target'] for w in W]
    files=[x for x in (a.input/'kernelslist.g').read_text().split() if x.startswith('kernel')][a.first:a.last+1]
    a.output.mkdir(parents=True,exist_ok=False); rep=[]
    for name in files:
        moved=0
        with lzma.open(a.input/name,'rt') as f, lzma.open(a.output/name,'wt',preset=1) as g:
            for line in f:
                t=line.split()
                if len(t)<7 or not re.fullmatch('[0-9a-fA-F]+',t[0]): g.write(line); continue
                oi=3+int(t[2]); wi=oi+2+int(t[oi+1]); width=int(t[wi])
                if not width: g.write(line); continue
                mode=int(t[wi+1]); idx=wi+2; n=int(t[1],16).bit_count()
                if mode==0: addr=[int(x,16) for x in t[idx:idx+n]]; end=idx+n
                elif mode==1: addr=[int(t[idx],16)+i*int(t[idx+1]) for i in range(n)]; end=idx+2
                elif mode==2:
                    addr=[int(t[idx],16)]
                    for d in t[idx+1:idx+n]: addr.append(addr[-1]+int(d))
                    end=idx+n
                else: raise ValueError('address mode')
                new=[]; changed=False
                for x in addr:
                    j=bisect.bisect_right(T,x)-1; w=W[j] if j>=0 else None
                    if w and w['target']<=x<w['target']+w['bytes'] and w['dest']!=w['target']:
                        new.append(w['dest']+x-w['target']); changed=True; moved+=1
                    else: new.append(x)
                if changed:
                    shifts={b-c for c,b in zip(addr,new)}
                    if mode in (1,2) and len(shifts)==1: t[idx]=hex(new[0])
                    else: t=t[:wi+1]+['0']+[hex(x) for x in new]+t[end:]
                    line=' '.join(t)+'\n'
                g.write(line)
        rep.append(dict(file=name,moved_lane_accesses=moved))
    (a.output/'kernelslist.g').write_text(''.join(x+'\n' for x in files))
    (a.output/'manifest.json').write_text(json.dumps(dict(policy=a.hbf,base=a.base,weights=W,kernels=rep,hbf_bytes=cur-a.base),indent=1)+'\n')
    print('kernels',len(files),'HBF-resident bytes',cur-a.base, 'moved lane accesses',sum(r['moved_lane_accesses'] for r in rep))
if __name__=='__main__': main()
