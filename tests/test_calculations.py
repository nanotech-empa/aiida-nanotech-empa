""" Tests for calculations

"""
import os
from . import TEST_DIR


def test_process(nanotech_empa_code):
    """Test running a calculation
    note this does not test that the expected outputs are created of output parsing"""
    from aiida.plugins import DataFactory, CalculationFactory
    from aiida.engine import run

    # Prepare input parameters
    DiffParameters = DataFactory('nanotech_empa')
    parameters = DiffParameters({'ignore-case': True})

    from aiida.orm import SinglefileData
    file1 = SinglefileData(
        file=os.path.join(TEST_DIR, "input_files", 'file1.txt'))
    file2 = SinglefileData(
        file=os.path.join(TEST_DIR, "input_files", 'file2.txt'))

    # set up calculation
    inputs = {
        'code': nanotech_empa_code,
        'parameters': parameters,
        'file1': file1,
        'file2': file2,
        'metadata': {
            'options': {
                'max_wallclock_seconds': 30
            },
        },
    }

    result = run(CalculationFactory('nanotech_empa'), **inputs)
    computed_diff = result['nanotech_empa'].get_content()

    assert 'content1' in computed_diff
    assert 'content2' in computed_diff
