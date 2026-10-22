from pathlib import Path
root=Path(__file__).parent
s=(root/'submit-two-gpu-feasibility.js').read_text(encoding='utf-8-sig')
s=s.replace("'scripts/launch_two_gpu_feasibility.sh','scripts/cpu_power_ema.py','status/cpu-ema-unit.log'","'scripts/launch_efficiency_sweep.sh','scripts/run_efficiency_sweep.py','scripts/async_cpu_ema.py','scripts/train_efficiency_worker.py','status/async-ema-unit.log','status/two-gpu-perf-repair-result.json'")
start=s.index('  const passed=');end=s.index('  const q=',start)
s=s[:start]+'''  const verified=await j.evaluate(async()=>{
    const base=location.pathname.split('/lab')[0]+'/api/contents/codex_remote_ops/flux_piperx_20261008/';
    const r=await fetch(base+'status/two-gpu-perf-repair-result.json?content=1&_='+Date.now());
    const proof=r.ok?JSON.parse((await r.json()).content):{};
    const u=await fetch(base+'status/async-ema-unit.log?content=1&_='+Date.now());
    return {proof,unit:u.ok&&(await u.json()).content.includes('ASYNC_DUAL_EMA_EXACT_SNAPSHOT_AND_CHECKPOINT_FLUSH_PASS')};
  });
  if(!verified.proof.save_resume_export||!verified.proof.candidate_valid||!verified.unit)return {submitted:false,reason:'Full prior repair gate or async equivalence missing',verified};
'''+s[end:]
start=s.index('      const found=');end=s.index('if(found)',start)
s=s[:start]+"      const found=a.find(x=>x.name===name&&!(((x.job_id==='job-REPLACE_WITH_YOUR_JOB_ID'&&x.status==='job_succeeded')||(x.job_id==='job-REPLACE_WITH_YOUR_JOB_ID'&&x.status==='job_failed'))&&x.command.endsWith('/flux_piperx_20261008/scripts/launch_perf_repair.sh')));"+s[end:]
s=s.replace('/scripts/launch_two_gpu_feasibility.sh','/scripts/launch_efficiency_sweep.sh')
s=s.replace('job-REPLACE_WITH_YOUR_JOB_ID','job-REPLACE_WITH_YOUR_JOB_ID')
s=s.replace("'150'","'180'").replace('max_runtime_minutes:150','max_runtime_minutes:180')
start=s.index("  await d.locator('#description').fill(");end=s.index('\n',start)
s=s[:start]+"  await d.locator('#description').fill('FLUX 2H100 efficiency sweep: preserved FP32 masters/reduce/dual EMA/global32. Compare buffered vs async CPU EMA, prefetch2/16, CPUthreads4/8, micro1/2(and4 only if2 fits). Select real wall throughput with GPU telemetry, finite loss/grad, then full save/resume/export and safe-stop tests. Formal tasks remain stopped pending measured selection.');"+s[end:]
(root/'submit-efficiency-sweep.js').write_text(s,encoding='utf-8')
