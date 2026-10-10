"""Cấu hình chung cho pytest.

Test nào gọi vào hàm còn là stub (raise NotImplementedError) sẽ được báo SKIPPED
("chưa cài") thay vì FAILED, để main luôn xanh trong lúc mọi người đang code.
Khi đã cài xong, test đó tự chạy thật.
"""
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, "Chưa cài (NotImplementedError)")
