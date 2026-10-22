from pathlib import Path
r=Path(__file__).parent;s=(r/'launch_flux_milestone_validation.sh').read_text()
s=s.replace('run_eval_paired.py','run_eval_curated.py').replace(' > "$FLUX_EVAL_LOG" 2>&1',' >> "$FLUX_EVAL_LOG" 2>&1')
(r/'launch_flux_milestone_validation_v2.sh').write_bytes(s.replace('\r\n','\n').encode())
