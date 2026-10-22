"""Bounded remote downloader: a failed file cannot hide progress in pool shutdown."""
from pathlib import Path
import concurrent.futures as cf
import fcntl,hashlib,json,os,time
from huggingface_hub import hf_hub_download,set_client_factory
import httpx
R=Path('/workspace/hdd/codex_remote_ops/flux_robodojo_20261013')
def bounded_results(items,worker,workers=4):
 with cf.ThreadPoolExecutor(max_workers=workers) as pool:
  pending={};iterator=iter(items)
  def fill():
   while len(pending)<2*workers:
    try:item=next(iterator)
    except StopIteration:break
    pending[pool.submit(worker,item)]=item
  fill()
  while pending:
   completed,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
   for future in completed:
    item=pending.pop(future)
    try:yield item,future.result(),None
    except Exception as error:yield item,None,type(error).__name__
   fill()
def main():
 lock=(R/'status/selected-download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 set_client_factory(lambda:httpx.Client(timeout=httpx.Timeout(120,connect=15),follow_redirects=True))
 plan=json.loads((R/'status/selected-download-plan.json').read_text());assert plan['revision']=='35efbc7dedfdbeeb6e95fb749bd885d73d483e41'
 assert plan['selected_episode_ids']=={'cover_blocks':list(range(300,400)),'match_and_pick_from_conveyor':list(range(1600,1700))}
 files=plan['files'];assert len(files)==15400 and len({f['path'] for f in files})==len(files)
 done={};failed={};status=R/'status/selected-download-progress.json';ledger=R/'status/selected-download-receipts.jsonl'
 def save():
  payload={'time':time.time(),'version':2,'total_files':len(files),'verified_files':len(done),'verified_bytes':sum(v['bytes'] for v in done.values()),'total_bytes':plan['bytes'],'failed_files':failed}
  temp=status.with_suffix('.tmp');temp.write_text(json.dumps(payload));temp.replace(status)
 def one(spec):
  relative=Path(spec['path']);assert not relative.is_absolute() and '..' not in relative.parts
  path=R/'official-source'/relative
  for attempt in range(3):
   try:
    if not path.exists():path=Path(hf_hub_download('RoboDojo-Benchmark/RoboDojo',spec['path'],repo_type='dataset',revision=plan['revision'],local_dir=R/'official-source'))
    if path.stat().st_size!=spec['bytes']:raise ValueError('size_mismatch')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if spec['lfs_sha256'] and digest!=spec['lfs_sha256']:raise ValueError('sha_mismatch')
    return {'bytes':spec['bytes'],'sha256':digest,'mtime_ns':path.stat().st_mtime_ns}
   except Exception as e:
    print(json.dumps({'event':'FILE_RETRY','path':spec['path'],'attempt':attempt+1,'error_type':type(e).__name__}),flush=True)
    if attempt==2:raise
    time.sleep(2**attempt)
 save()
 with ledger.open('a') as receipt:
  for spec,result,error in bounded_results(files,one):
   if error:failed[spec['path']]=error
   else:
    done[spec['path']]=result;receipt.write(json.dumps({'path':spec['path'],**result})+'\n');receipt.flush()
   save()
 if failed:
  print(json.dumps({'event':'INCOMPLETE','failed_files':failed}),flush=True);raise SystemExit(1)
 (R/'status/selected-download-verified.json').write_text(json.dumps({'time':time.time(),'complete':True,'revision':plan['revision'],'selected_episode_ids':plan['selected_episode_ids'],'files':done},indent=2))
 assets=R/'RoboDojo/Assets'
 if not assets.exists():assets.symlink_to(R/'official-source/Assets',target_is_directory=True)
 print('SELECTED_DATA_AND_ASSETS_VERIFIED',flush=True)
if __name__=='__main__':main()
