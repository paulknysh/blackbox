import logging
from .utils import f, run_search_min, get_bb_logger

def test_logger_info(caplog):
    """
    Test that logger INFO is printed when level is set to INFO
    """
    get_bb_logger().setLevel(logging.INFO)
    # This call should generate 3 logs. 2 logs for running and 1 log for DONE
    run_search_min(f, [[-10,10]], 10, 5, None)
    logs = 0
    with caplog.at_level(logging.DEBUG):
        for i, record in enumerate(caplog.records):
            logs += 1
            assert record.levelno == logging.INFO
            if i == 2:
                assert record.message.startswith('DONE: @') 
            else:
                assert record.message.startswith('evaluating batch {}/2'.format(i+1))
    assert logs == 3 

def test_logger_notset(caplog):
    """
    Test that no output is printed when logger is set to NOTSET
    """
    get_bb_logger().setLevel(logging.NOTSET)
    run_search_min(f, [[-10,10]], 10, 5, None)
    with caplog.at_level(logging.DEBUG):
        for _ in caplog.records:
            assert False

def test_logger_info_budget_adgustment(caplog):
    """
    Test that logger.info is printed if function_calls % budget != 0
    """
    get_bb_logger().setLevel(logging.INFO)
    run_search_min(f, [[-10,10]], 10, 4, None)
    logs = 0
    with caplog.at_level(logging.INFO):
        assert len(caplog.records) > 0
        assert caplog.records[0].levelno == logging.INFO
        assert caplog.records[0].message.startswith('budget was adjusted to be ')

def test_logger_error_insufficient_budget(caplog):
    """
    Test that logger.error is printed if insufficient budget
    """
    get_bb_logger().setLevel(logging.ERROR)
    run_search_min(f, [[-10,10]], 1, 1, None)
    logs = 0
    with caplog.at_level(logging.ERROR):
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR
        assert caplog.records[0].message.startswith('budget is not sufficient')
