from pathlib import Path
r=Path(__file__).parent;s=(r/'run_eval_recorded.py').read_text(encoding='utf-8-sig')
key="(out/'protocol.json').write_text(json.dumps(args,indent=2))"
replacement='''reference_path=B/'evaluation'/args['task_name']/'flux3000-validation20/paired_seeds.json'
reference=json.loads(reference_path.read_text()) if args['seed']==8 and args['test_num']==20 else None
if reference:
    expected={x['seed']:x['instruction'] for x in reference}
    original_instruction=upstream.build_instruction
    def paired_instruction(config,episode_info,instruction_type,test_num,seed):
        if seed not in expected:raise RuntimeError(f'Paired validation seed {seed} differs from reference; investigate expert filter')
        return expected[seed]
    upstream.build_instruction=paired_instruction
args['paired_reference']=str(reference_path) if reference else None
(out/'protocol.json').write_text(json.dumps(args,indent=2))'''
s=s.replace(key,replacement)
s=s.replace("(out/'paired_seeds.json').write_text(json.dumps(paired,indent=2))","if reference:\n    assert [(x['seed'],x['instruction']) for x in paired]==[(x['seed'],x['instruction']) for x in reference], 'Paired cases changed'\n(out/'paired_seeds.json').write_text(json.dumps(paired,indent=2))")
(r/'run_eval_paired.py').write_text(s,encoding='utf-8')
p=r/'launch_flux_milestone_validation.sh';p.write_text(p.read_text().replace('run_eval_recorded.py','run_eval_paired.py'),encoding='utf-8')
