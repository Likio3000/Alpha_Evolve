import logging
from utils.log_counter import CountingHandler

def test_counting_handler_counts_levels_and_messages(tmp_path):
    logger = logging.getLogger('lc')
    logger.setLevel(logging.DEBUG)
    handler = CountingHandler()
    logger.addHandler(handler)

    logger.info('hello')
    logger.warning('hello')
    logger.error('bye')

    counts = handler.get_counts()
    assert counts['levels']['INFO'] == 1
    assert counts['levels']['WARNING'] == 1
    assert counts['levels']['ERROR'] == 1
    assert counts['messages']['hello'] == 2
    assert counts['messages']['bye'] == 1

    summary = tmp_path / 'summary.txt'
    handler.dump_summary(summary)
    text = summary.read_text()
    assert 'INFO: 1' in text
    assert 'bye: 1' in text

    logger.removeHandler(handler)
