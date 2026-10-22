import io,json,pathlib,sys,zipfile
import h5py,numpy as np
B=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B));sys.path.insert(0,str(B/'scripts'))
from XPolicyLab.utils.process_data import decode_image_bit
from convert_robotwin import video_write,JOINTS,CAMERAS
archive=next(p for p in (B/'assets').rglob('demo_clean.zip') if 'click_bell' in str(p))
with zipfile.ZipFile(archive) as z:
    name=next(n for n in z.namelist() if n.endswith('.hdf5'))
    with h5py.File(io.BytesIO(z.read(name))) as h:
        fps=int(h['additional_info/frequency'][()])
        raw=h['instructions'][()]
        captions=json.loads(raw.decode())
        state=np.concatenate([h[f'state/{k}'][()] for k in JOINTS],axis=1)
        action=np.concatenate([h[f'action/{k}'][()] for k in JOINTS],axis=1)
        assert state.shape==action.shape and state.shape[1]==14
        assert np.isfinite(action).all() and np.isfinite(state).all()
        result={'episode':name,'fps':fps,'shape':list(state.shape),'caption_type':type(captions).__name__,
                'caption_preview':str(captions)[:500],'cameras':{},'state_action_max_abs_difference':float(np.abs(state-action).max())}
        for c in CAMERAS:
            frames=decode_image_bit(h[f'vision/{c}/colors'][()][:5])
            video_write(B/'status'/f'smoke-{c}.mp4',frames,fps)
            result['cameras'][c]={'shape':list(frames.shape),'lossless_rgb_roundtrip':True}
(B/'status/data-smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
