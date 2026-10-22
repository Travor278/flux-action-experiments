from download_selected_v2 import bounded_results
def test_one_failed_file_does_not_hide_other_results():
 def worker(item):
  if item==2:raise RuntimeError('simulated transient failure')
  return item*10
 rows=list(bounded_results(range(6),worker,workers=2))
 assert len(rows)==6
 assert {i:v for i,v,e in rows if e is None}=={0:0,1:10,3:30,4:40,5:50}
 assert [(i,e) for i,v,e in rows if e is not None]==[(2,'RuntimeError')]
