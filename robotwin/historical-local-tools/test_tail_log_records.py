from tail_log_records import log_dicts

def test_scalar_nccl_and_concatenated_rank_events():
    text='1\ntrue\nnull\n[Gloo] Rank 1 connected\n'
    text+='{"event":"EFFICIENCY_RANK_READY","rank":0,"step":4}{"event":"EFFICIENCY_RANK_READY","rank":1,"step":4}\n'
    assert [r['rank'] for r in log_dicts(text)]==[0,1]

def test_prefixed_objects_and_partial_lines_do_not_fabricate_events():
    text='prefix {garbage} {"event":"other"}\n{"event":"broken"\n2\n[]\n'
    assert list(log_dicts(text))==[{'event':'other'}]
