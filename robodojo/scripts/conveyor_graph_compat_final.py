"""Restore NVIDIA's declared variable-to-conveyor link in composed runtime USD.

Does not write the packaged asset or change its declared speed. Installed
IsaacSim5.1 CreateConveyorBelt.commands.py uses this exact connection.
"""
import json,time

def restore_loaded_conveyor_graph(stage,receipt_path,phase='post_reset'):
    import omni.graph.core as og
    from pxr import Usd
    records=[]
    for node in Usd.PrimRange(stage.GetPseudoRoot()):
        attr=node.GetAttribute('node:type')
        if not attr or attr.Get()!='isaacsim.asset.gen.conveyor.IsaacConveyor':continue
        graph=node.GetParent();reader=graph.GetChild('read_speed')
        assert reader.GetAttribute('inputs:variableName').Get()=='Velocity'
        declared=graph.GetAttribute('graph:variable:Velocity');speed=float(declared.Get())
        assert abs(speed+0.1)<1e-6
        source=reader.GetAttribute('outputs:value');dest=node.GetAttribute('inputs:velocity');expected=source.GetPath()
        previous=dest.GetConnections()
        native=graph.GetChild('read_speed_compat_5_1')
        if native and previous==[native.GetAttribute('outputs:value').GetPath()]:
            assert native.GetAttribute('node:type').Get()=='omni.graph.core.ReadVariable'
            assert native.GetAttribute('inputs:variableName').Get()=='Velocity'
            expected=native.GetAttribute('outputs:value').GetPath()
        if previous and previous!=[expected]:raise RuntimeError(str(previous))
        if not previous:
            assert stage.GetEditTarget().GetLayer().identifier.startswith('anon:')
            connected=og.Controller.connect(str(expected),str(dest.GetPath()),update_usd=True)
            if dest.GetConnections()!=[expected]:
                # The asset's legacy ReadVariable output is authored as token;
                # native5.1 declares it any and resolves it against float input.
                # Recreate ONLY that reader using NVIDIA's native constructor
                # pattern, preserving the graph variable and existing actuator.
                assert str(source.GetTypeName())=='token' and str(dest.GetTypeName())=='float'
                name='read_speed_compat_5_1';keys=og.Controller.Keys
                edit={keys.SET_VALUES:[(f'{name}.inputs:graph',str(graph.GetPath())),(f'{name}.inputs:variableName','Velocity')],keys.CONNECT:[(str(graph.GetPath())+f'/{name}.outputs:value',str(dest.GetPath()))]}
                if not graph.GetChild(name):edit[keys.CREATE_NODES]=[(name,'omni.graph.core.ReadVariable')]
                existing_graph=og.Controller.graph(str(graph.GetPath()))
                assert existing_graph.is_valid()
                og.Controller.edit(existing_graph,edit)
                expected=graph.GetChild(name).GetAttribute('outputs:value').GetPath()
                with receipt_path.open('a') as stream:stream.write(json.dumps({'time':time.time(),'phase':'native_reader_recreated','source_type':str(source.GetTypeName()),'destination_type':str(dest.GetTypeName()),'direct_connect_result':str(connected),'declared_speed_unchanged':speed})+'\n')
        assert dest.GetConnections()==[expected] and float(declared.Get())==speed
        records.append({'time':time.time(),'phase':phase,'source':str(expected),'destination':str(dest.GetPath()),'restored_missing_link':not previous,'declared_speed_before':speed,'declared_speed_after':float(declared.Get())})
    assert len(records)==1
    with receipt_path.open('a') as stream:
        for record in records:stream.write(json.dumps(record)+'\n')

def install_conveyor_graph_compat(receipt_path):
    import omni.graph.core as og
    from pxr import Usd
    from env.scene_manager.objects.dynamic import DynamicObject
    original=DynamicObject.__init__
    if getattr(original,'_flux_conveyor_compat',False):raise RuntimeError('Already installed')
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        if self.model_name!='conveyor':return
        records=[]
        for node in Usd.PrimRange(self.stage.GetPrimAtPath(self.usd_prim_path)):
            attr=node.GetAttribute('node:type')
            if not attr or attr.Get()!='isaacsim.asset.gen.conveyor.IsaacConveyor':continue
            graph=node.GetParent();reader=graph.GetChild('read_speed')
            assert reader.GetAttribute('node:type').Get()=='omni.graph.core.ReadVariable'
            assert reader.GetAttribute('inputs:variableName').Get()=='Velocity'
            declared=graph.GetAttribute('graph:variable:Velocity');speed=float(declared.Get())
            assert abs(speed+0.1)<1e-6,speed
            source=reader.GetAttribute('outputs:value');dest=node.GetAttribute('inputs:velocity')
            expected=source.GetPath();previous=dest.GetConnections()
            if previous and previous!=[expected]:raise RuntimeError(f'Unexpected conveyor input {previous}')
            repaired=not previous
            if repaired:
                assert self.stage.GetEditTarget().GetLayer().identifier.startswith('anon:')
                og.Controller.connect(str(source.GetPath()),str(dest.GetPath()),update_usd=True)
            assert dest.GetConnections()==[expected]
            assert float(declared.Get())==speed
            records.append({'time':time.time(),'source':str(source.GetPath()),'destination':str(dest.GetPath()),'declared_speed_before':speed,'declared_speed_after':float(declared.Get()),'restored_missing_link':repaired})
        assert len(records)==1,records
        with receipt_path.open('a') as stream:
            for record in records:stream.write(json.dumps(record)+'\n')
    initialize._flux_conveyor_compat=True
    DynamicObject.__init__=initialize
