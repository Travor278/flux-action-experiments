from pathlib import Path
root=Path(__file__).parent
source=(root/'submit-two-gpu-feasibility.js').read_text(encoding='utf-8-sig')
for task,version in [('click_bell',2),('handover_block',3)]:
 s=source.replace("const name='v2sam-pr7-matched-production-primary-v1';",f"const name='v2sam-pr7-matched-production-primary-v{version}';")
 s=s.replace("'scripts/launch_two_gpu_feasibility.sh','scripts/cpu_power_ema.py','status/cpu-ema-unit.log'",f"'scripts/launch_train_10k.sh','scripts/verify_10k_export.py','configs/{task}-10k-2gpu.json','status/ten-k-ready.json','status/two-gpu-feasibility-verified.json'")
 start=s.index('  const passed=');end=s.index('  const q=',start)
 s=s[:start]+'''  const verified=await j.evaluate(async()=>{
    const base=location.pathname.split('/lab')[0]+'/api/contents/codex_remote_ops/flux_piperx_20261008/';
    const records=[];for(const file of ['status/two-gpu-feasibility-verified.json','status/ten-k-ready.json']){
      const r=await fetch(base+file+'?content=1&_='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error('Missing gate');records.push(JSON.parse((await r.json()).content));
    }return records;
  });
  if(!verified.every(x=>x.verified)||!verified[0].save_resume_export||verified[0].world_size!==2||verified[0].global_batch!==32)throw Error('2GPU or 10k gate invalid');
'''+s[end:]
 start=s.index('      const found=');end=s.index('if(found)',start)
 s=s[:start]+"      const found=a.find(x=>x.name===name);"+s[end:]
 s=s.replace('/scripts/launch_two_gpu_feasibility.sh',f'/scripts/launch_train_10k.sh {task}')
 s=s.replace('job-REPLACE_WITH_YOUR_JOB_ID','job-REPLACE_WITH_YOUR_JOB_ID')
 s=s.replace('#flux-two-gpu-submit',f'#flux-10k-{task}-submit')
 start=s.index("  await d.locator('#description').fill(");end=s.index('\n',start)
 desc=f'FLUX {task} independent 10000-update SFT from official base; 50 demo_clean; 2 H100, micro1 x16accum global32; FP32 master/dual CPU FP32 EMA/every microbatch sync validated with full save/resume/export. LR trunk1e-5 heads5e-5; freeze200 warm600/200 cooldown9000-10000. Milestone EMA exports1k/3k/5k/10k; native keep1 resumable checkpoint.'
 s=s[:start]+f"  await d.locator('#description').fill('{desc}');"+s[end:]
 s=s.replace("'150'","'10080'").replace('max_runtime_minutes:150','max_runtime_minutes:10080')
 (root/f'submit-10k-{task}.js').write_text(s,encoding='utf-8')
