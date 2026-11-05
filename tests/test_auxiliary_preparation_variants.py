"""CPU-only frozen-auxiliary selection tests; no CUDA, weights, or pixels."""
import ast
from contextlib import ExitStack
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vsr import train


_ABSENT = object()
_BASELINE_NUMERIC_AST = {
    "make_model": "8aac38fccf957c76c90fe769aee0b636b794810ceb68691d94e94ae902dd56b6",
    "train_variant": "3bf471a4ae1f21869c5fde65a9d5de33813ec2c9926506fbaf66f67d093bee6e",
    "main": "1c09dba4d4c49ab5ff42c6646399747d4bddb3ad99ea64f391adf3a35624ebe6",
}


def _ast_structure(value):
    # ast.dump's empty-field defaults changed after Python3.12; normalize only
    # absent optional/empty fields while preserving node types and values.
    if isinstance(value,ast.AST):
        return [type(value).__name__,[[key,_ast_structure(v)] for key,v in ast.iter_fields(value)
                if v is not None and v!=[]]]
    if isinstance(value,list):return [_ast_structure(v) for v in value]
    return value


class AuxiliaryPreparationVariants(unittest.TestCase):
    def config(self, trained=("Base","Align"), preparation=_ABSENT):
        config = {"training": {"variants": list(trained)},
            "teacher": {"source_dir":"vendor/fixture", "weights":"weights/fixture_teacher.pth",
                "weights_sha256":"a"*64, "revision":"fixture_revision"},
            "flow": {"weights":"weights/fixture_flow.pth", "weights_sha256":"b"*64,"updates":12}}
        if preparation is not _ABSENT:
            config["training"]["preparation_variants"] = preparation
        return config

    def call(self, config):
        self.events, self.dumps, self.saved = [], {}, {}
        events = self.events
        class Input:
            def to(self, device):
                events.append(("input_to",device))
                return self
        class Teacher:
            def __init__(self,*args):
                events.append(("teacher_constructor",tuple(map(str,args))))
            def to(self,device):
                events.append(("teacher_to",device));return self
            def __call__(self,hr,anchors,grid):
                events.append(("teacher_forward",tuple(anchors),tuple(grid)))
                return torch.tensor([[[.25,.75]]])
        class Flow:
            def __init__(self,*args,**kwargs):
                events.append(("flow_constructor",tuple(map(str,args)),kwargs))
                self.provenance={"source":"CPUfixture"}
            def to(self,device):
                events.append(("flow_to",device));return self
            def estimate_anchor_flows(self,hr,anchors):
                events.append(("flow_forward",tuple(anchors)))
                return torch.tensor([[[[1.0]]]]),torch.tensor([[[[-1.0]]]])
        def dump(path,value):
            self.dumps[Path(path).name] = copy.deepcopy(value)
        def save(value,path):
            self.saved[Path(path).name] = copy.deepcopy(value)
        clock=itertools.count()
        with ExitStack() as stack:
            stack.enter_context(patch.object(train,"DinoTeacher",Teacher))
            stack.enter_context(patch.object(train,"FlowTeacher",Flow))
            stack.enter_context(patch.object(train.torch.cuda,"synchronize",lambda:events.append(("sync",))))
            stack.enter_context(patch.object(train,"release",lambda:events.append(("release",))))
            stack.enter_context(patch.object(train,"dump",dump))
            stack.enter_context(patch.object(train.torch,"save",save))
            stack.enter_context(patch.object(train,"file_hash",lambda path:"c"*64))
            stack.enter_context(patch.object(train.time,"perf_counter",lambda:float(next(clock))))
            return train.prepare_auxiliary(config,[{"hr":Input()}],
                {"anchors":[0,4,8],"token_grid":[3,8,14]},Path("/unused_CPUfixture"))

    def test_HRI_Base_Align_can_prepare_all_frozen_teacher_and_flow_targets(self):
        config=self.config(preparation=["Base","Align","Relation","Both"])
        config["reference"]={"enabled":True}
        before=copy.deepcopy(config)
        features,flows=self.call(config)
        self.assertEqual(config,before)
        self.assertEqual(config["training"]["variants"],["Base","Align"])
        self.assertIsNotNone(features);self.assertIsNotNone(flows)
        self.assertEqual(set(self.saved),{"fresh_teacher_features.pt","fresh_anchor_flows.pt"})
        self.assertEqual(set(self.dumps),{"teacher_cache_contract.json","flow_provenance.json","auxiliary_preprocessing_cost.json"})
        self.assertEqual(sum(event[0]=="teacher_constructor" for event in self.events),1)
        self.assertEqual(sum(event[0]=="flow_constructor" for event in self.events),1)
        self.assertTrue(torch.equal(flows[0]["forward"],torch.tensor([[[[1.0]]]])))

    def test_absent_field_keeps_legacy_Base_and_Align_preparation(self):
        features,flows=self.call(self.config(trained=("Base",)))
        self.assertIsNone(features);self.assertIsNone(flows)
        self.assertEqual(self.events,[])
        self.assertEqual(self.saved,{})
        features,flows=self.call(self.config(trained=("Base","Align")))
        self.assertIsNotNone(features);self.assertIsNone(flows)
        self.assertEqual(set(self.saved),{"fresh_teacher_features.pt"})
        self.assertFalse(any(event[0]=="flow_constructor" for event in self.events))

    def test_explicit_identical_set_matches_legacy_calls_and_payloads(self):
        variants=["Base","Align","Relation","Both"]
        old=self.call(self.config(trained=variants))
        events=copy.deepcopy(self.events);dumps=copy.deepcopy(self.dumps);saved=copy.deepcopy(self.saved)
        new=self.call(self.config(trained=variants,preparation=list(variants)))
        self.assertEqual(self.events,events)
        self.assertEqual(self.dumps,dumps)
        for left,right in zip(old[0],new[0]):self.assertTrue(torch.equal(left,right))
        for left,right in zip(old[1],new[1]):
            for direction in ["forward","backward"]:self.assertTrue(torch.equal(left[direction],right[direction]))
        for path in saved:
            self.assertEqual(len(saved[path]),len(self.saved[path]))

    def test_invalid_or_missing_superset_rejected_before_any_GPU_operation(self):
        invalid=[None,[],"Both",["Base","Align","Unknown"],["Base","Align","Align"],["Base"],
                 ["Align"],True,["Base",{"name":"Align"}]]
        for preparation in invalid:
            with self.subTest(preparation=preparation):
                config=self.config(preparation=preparation);before=copy.deepcopy(config)
                with self.assertRaisesRegex(ValueError,"training.preparation_variants"):
                    self.call(config)
                self.assertEqual(self.events,[]);self.assertEqual(self.dumps,{});self.assertEqual(self.saved,{})
                self.assertEqual(config,before)
        for trained in [[],["Unknown"],["Base","Base"],"Both",True]:
            with self.subTest(trained=trained):
                config=self.config();config["training"]["variants"]=trained
                with self.assertRaisesRegex(ValueError,"training.variants"):
                    self.call(config)
                self.assertEqual(self.events,[])

    def test_validation_does_not_draw_RNG_or_mutate_tuple_inputs(self):
        config=self.config();config["training"]["variants"]=("Base","Align")
        config["training"]["preparation_variants"]=("Base","Align","Relation","Both")
        before=copy.deepcopy(config);py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state().clone()
        self.call(config)
        self.assertEqual(config,before)
        self.assertEqual(random.getstate(),py)
        now=np.random.get_state()
        self.assertEqual(now[0],npstate[0]);self.assertTrue(np.array_equal(now[1],npstate[1]));self.assertEqual(now[2:],npstate[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),cpu))

    def test_numeric_model_training_and_main_AST_match_prechange_production(self):
        tree=ast.parse(Path(train.__file__).read_text())
        functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        for name,expected in _BASELINE_NUMERIC_AST.items():
            actual=hashlib.sha256(json.dumps(_ast_structure(functions[name]),separators=(',',':')).encode()).hexdigest()
            self.assertEqual(actual,expected,name)


if __name__=="__main__":
    unittest.main()
