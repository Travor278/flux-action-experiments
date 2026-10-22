"""Regression: a stop must escape Trainer's outer epoch loop after one save."""
import ast,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
class SafeStop(Exception):pass
def run(name):
    with tempfile.TemporaryDirectory(prefix='stop-control-test-',dir=R/'status') as directory:
        out=Path(directory);(out/'STOP_AFTER_UPDATE').touch()
        t=SimpleNamespace(step=717,_written=717,rank=0,device='cpu',output_dir=out,saves=0)
        def checkpoint():t.saves+=1
        t._checkpoint=checkpoint
        def original(*a,**kw):t.step+=1;return False
        torch=SimpleNamespace(tensor=lambda x,**kw:SimpleNamespace(item=lambda:x),distributed=SimpleNamespace(broadcast=lambda *a,**kw:None))
        ns={'t':t,'original':original,'torch':torch,'SafeStop':SafeStop,'AsyncCPUPowerEMA':SimpleNamespace(flush=lambda:None),'json':json,'time':time}
        tree=ast.parse((R/'scripts'/name).read_text())
        update=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='update')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[update],type_ignores=[])),name,'exec'),ns)
        escaped=False
        try:
            for _ in range(2):
                # Returning true only leaves the inner batch loop; outer epoch
                # condition is still step < configured 10000, so it repeats.
                for _ in range(5):
                    if ns['update']():break
        except SafeStop:escaped=True
        return {'step':t.step,'saves':t.saves,'escaped':escaped,'saved_step':json.loads((out/'STOPPED_SAFELY.json').read_text())['step']}
old=run('train_formal.py');new=run('train_formal_stopfixed.py')
assert old['step']==719 and old['saves']==2 and not old['escaped']
assert new=={'step':718,'saves':1,'escaped':True,'saved_step':718}
(R/'status/safe-stop-outer-loop-regression.json').write_text(json.dumps({'time':time.time(),'verified':True,'original':old,'fixed':new,'scope':'Control-flow regression only; production training math/config/source remain unchanged'},indent=2))
print('SAFE_STOP_OUTER_LOOP_REGRESSION_PASS')
