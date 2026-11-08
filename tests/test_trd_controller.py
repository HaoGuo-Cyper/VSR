"""Prospective TRD calibration and metadata guard regressions; CPU only."""
import copy
import json
import math
import unittest
from vsr import train as worker


def rows():
    return [dict(data_index=i,nominal_sigma=sigma,fm=1.,unit_aux=.1,
                 fm_grad_norm=2.,aux_grad_norm=1.,head_grad_norm=.5)
            for i in range(5) for sigma in (.2,.5,.8)]


class TRDCalibrationTests(unittest.TestCase):
    def test_one_common_train_only_weight_uses_median_and_two_significant_digits(self):
        data=rows();data[-1]['fm_grad_norm']=200.
        result=worker._trd_calibration_weight(data)
        self.assertTrue(result['eligible']);self.assertEqual(result['weight'],.2)
        self.assertEqual(result['positive_gradient_cases'],15)
        self.assertFalse(result['selection_uses_validation_or_test'])
        self.assertEqual(result['optimizer_updates'],0)

    def test_finite_zero_gradients_are_reported_and_counted_without_inventing_ratios(self):
        data=rows()
        for row in data[8:]:row['aux_grad_norm']=0.
        result=worker._trd_calibration_weight(data)
        self.assertTrue(result['eligible']);self.assertEqual(result['positive_gradient_cases'],8)
        self.assertEqual(len(result['rows']),15)
        self.assertEqual(result['weight'],.2)

    def test_seven_positive_cases_stop_and_do_not_supply_weight(self):
        data=rows()
        for row in data[7:]:row['aux_grad_norm']=0.
        result=worker._trd_calibration_weight(data)
        self.assertFalse(result['eligible']);self.assertIsNone(result['weight'])

    def test_any_nonfinite_or_negative_observation_fails_whole_matrix(self):
        for key in ('fm','unit_aux','fm_grad_norm','aux_grad_norm','head_grad_norm'):
            for invalid in (float('nan'),float('inf'),-1.):
                data=rows();data[-1][key]=invalid
                with self.subTest(key=key,value=invalid),self.assertRaises(ValueError):
                    worker._trd_calibration_weight(data)

    def test_matrix_cannot_drop_or_reorder_training_examples(self):
        data=rows()
        for changed in (data[:-1],list(reversed(data))):
            with self.assertRaises(ValueError):worker._trd_calibration_weight(changed)

    def test_cli_keeps_explicit_prepare_and_execute_boundaries(self):
        parser=worker.build_parser()
        value=parser.parse_args(['trd-prepare','--config','study.json','--run-dir','new'])
        self.assertEqual(value.command,'trd-prepare')
        with self.assertRaises(SystemExit):parser.parse_args(['trd-preflight','--run-dir','new'])
        value=parser.parse_args(['trd-execute','--run-dir','new','--expected-protocol-sha','a'*64,'--resume'])
        self.assertTrue(value.resume)

if __name__=='__main__':unittest.main()
