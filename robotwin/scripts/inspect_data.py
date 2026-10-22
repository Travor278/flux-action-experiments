import io,json,pathlib,zipfile
import h5py
B=pathlib.Path('/workspace/hdd/codex_remote_ops/flux_piperx_20261008')
result={}
for zpath in (B/'assets').rglob('demo_clean.zip'):
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist(); sample=next(x for x in names if x.endswith('.hdf5'))
        obj={'sample':sample,'names':names[:12],'files':len(names),'fields':{}}
        with h5py.File(io.BytesIO(z.read(sample))) as f:
            def visit(n,d):
                if isinstance(d,h5py.Dataset):
                    obj['fields'][n]={'shape':list(d.shape),'dtype':str(d.dtype)}
            f.visititems(visit)
        for name in names:
            if name.endswith('.json'):
                obj['instruction_example']=json.loads(z.read(name));break
        result[str(zpath)]=obj
(B/'status/data-schema.json').write_text(json.dumps(result,indent=2))
