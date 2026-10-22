"""Fetch pinned official robot/object assets for the clean evaluation protocol."""
import hashlib,json,os,pathlib,zipfile
from huggingface_hub import HfApi,snapshot_download
B=pathlib.Path(__file__).resolve().parents[1]
REV='3dc3b798668feb99ac61cc9086d84cbcc3d79186'
final=B/'assets/robotwin-sim-assets'
marker=B/'assets/robotwin-sim-assets.manifest.json'
if marker.exists():raise SystemExit('Previously verified simulation assets')
tmp=B/'assets/.robotwin-sim-assets.download'
info=HfApi().repo_info('TianxingChen/RoboTwin2.0',repo_type='dataset',revision=REV,files_metadata=True)
files={x.rfilename:x for x in info.siblings}
snapshot_download('TianxingChen/RoboTwin2.0',repo_type='dataset',revision=REV,
                  allow_patterns=['embodiments.zip','objects.zip'],local_dir=tmp,max_workers=2)
records=[]
for name in ['embodiments.zip','objects.zip']:
    p=tmp/name;metadata=files[name]
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert p.stat().st_size==metadata.size and digest==metadata.lfs.sha256
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        for n in z.namelist():
            if not (tmp/n).resolve().is_relative_to(tmp.resolve()):raise ValueError('Unsafe ZIP path')
        z.extractall(tmp)
    records.append({'file':name,'bytes':p.stat().st_size,'sha256':digest})
for name in ['embodiments','objects']:
    if not (tmp/name).is_dir():raise ValueError('Unexpected asset archive layout: '+name)
tmp.rename(final)
for name in ['embodiments','objects']:
    link=B/'RoboTwin/assets'/name
    if link.exists():raise FileExistsError('Refusing to replace existing simulator assets: '+str(link))
    link.symlink_to(final/name,target_is_directory=True)
marker.write_text(json.dumps({'repo':'TianxingChen/RoboTwin2.0','revision':REV,'files':records,
    'scope':'clean background only; randomized background textures not installed'},indent=2))
print('SIMULATION_ASSETS_VERIFIED',flush=True)
