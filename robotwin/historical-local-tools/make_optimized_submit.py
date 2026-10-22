from pathlib import Path
r=Path(__file__).parent
for task in ['click_bell','handover_block']:
 s=(r/f'submit-10k-{task}.js').read_text(encoding='utf-8-sig')
 s=s.replace('scripts/launch_train_10k.sh','scripts/launch_train_optimized.sh').replace('scripts/verify_10k_export.py','scripts/verify_optimized_export.py').replace(f'configs/{task}-10k-2gpu.json',f'configs/{task}-10k-opt-v1.json').replace('status/ten-k-ready.json','status/optimized-ten-k-ready.json').replace('status/two-gpu-feasibility-verified.json','status/efficiency-selected-verified.json')
 s=s.replace("if(!verified.every(x=>x.verified)||!verified[0].save_resume_export||verified[0].world_size!==2||verified[0].global_batch!==32)throw Error('2GPU or 10k gate invalid');","if(!verified[0].complete||!verified[0].save_resume_export||!verified[0].safe_stop_verified||!verified[1].verified||verified[0].selected.micro!==4||verified[0].selected.ema_mode!=='async')throw Error('Optimized selection gate invalid');")
 s=s.replace('job-REPLACE_WITH_YOUR_JOB_ID','job-REPLACE_WITH_YOUR_JOB_ID')
 start=s.index("  await d.locator('#description').fill(");end=s.index('\n',start)
 s=s[:start]+f"  await d.locator('#description').fill('FLUX {task} optimized full10000-update SFT, officialbase/freshoptimizer/50demo_clean. Verified2H100/micro4/accum4/global32/asyncCPUdualFP32EMA/prefetch16/CPUthreads4. 10.64sec/update,97/94percent utilization measured. Firstcheckpoint100 then nativekeep1/1000; EMAexports1k3k5k10k. Exactresume and safe-stopcheckpoint verified.');"+s[end:]
 s=s.replace("'10080'","'4320'").replace('max_runtime_minutes:10080','max_runtime_minutes:4320')
 (r/f'submit-optimized-{task}.js').write_text(s,encoding='utf-8')
