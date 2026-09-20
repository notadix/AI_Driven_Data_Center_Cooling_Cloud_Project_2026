"""Timestream must degrade to the in-memory store when the cloud errors, not go blind."""

import logging
import time

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from database.timestream_client import TimestreamClient


def _payload(facility="DC-EAST-01", crac="CRAC-01", inlet=22.0, pue=1.05):
    return {"facility_id": facility, "crac_id": crac, "rack_id": "RACK-A01",
            "server_inlet_temp_c": inlet, "pue": pue, "it_power_mw": 0.018}


class _DownWrite:
    def __init__(self):
        self.calls = 0

    def write_records(self, **_):
        self.calls += 1
        raise ClientError({"Error": {"Code": "InternalFailure", "Message": "not yet implemented or pro feature"}}, "WriteRecords")


class _DownQuery:
    def __init__(self):
        self.calls = 0

    def query(self, **_):
        self.calls += 1
        raise ClientError({"Error": {"Code": "InternalFailure", "Message": "pro feature"}}, "Query")

    def get_paginator(self, _):
        outer = self

        class P:
            def paginate(self, **_):
                outer.calls += 1
                raise EndpointConnectionError(endpoint_url="http://localhost:4566")
        return P()


@pytest.fixture()
def cloud_client():
    c = TimestreamClient(local_mode=False)
    c._write_client, c._query_client = _DownWrite(), _DownQuery()
    return c


class TestWritesWhenCloudIsDown:
    def test_first_failure_trips_the_breaker_and_later_writes_skip_the_cloud(self, cloud_client):
        for _ in range(20):
            cloud_client.write_telemetry(_payload())
        assert cloud_client._write_client.calls == 1        # not one failing call per record
        assert cloud_client._cloud_failures == 1

    def test_warning_is_logged_once_not_per_record(self, cloud_client, caplog):
        with caplog.at_level(logging.WARNING):
            for _ in range(50):
                cloud_client.write_telemetry(_payload())
        assert len([r for r in caplog.records if "Timestream unavailable" in r.message]) == 1
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_records_are_never_lost(self, cloud_client):
        for i in range(10):
            cloud_client.write_telemetry(_payload(inlet=20.0 + i))
        assert len(cloud_client._store._records) == 10

    def test_cloud_is_retried_after_the_retry_window(self, cloud_client):
        cloud_client.CLOUD_RETRY_S = 0.05
        cloud_client.write_telemetry(_payload())
        time.sleep(0.1)
        cloud_client.write_telemetry(_payload())
        assert cloud_client._write_client.calls == 2

    def test_rejected_records_do_not_trip_the_breaker(self):
        class Rejecting:
            def write_records(self, **_):
                raise ClientError({"Error": {"Code": "RejectedRecordsException", "Message": "bad"}}, "WriteRecords")
        c = TimestreamClient(local_mode=False)
        c._write_client, c._query_client = Rejecting(), _DownQuery()
        c.write_telemetry(_payload())
        assert c._cloud_failures == 0 and c._cloud_ready()

    def test_recovery_resumes_cloud_writes(self, cloud_client):
        cloud_client.CLOUD_RETRY_S = 0.05
        cloud_client.write_telemetry(_payload())
        good_calls = []
        cloud_client._write_client = type("Ok", (), {"write_records": lambda self, **k: good_calls.append(1)})()
        time.sleep(0.1)
        cloud_client.write_telemetry(_payload())
        assert good_calls == [1] and cloud_client._cloud_failures == 0


class TestReadsWhenCloudIsDown:
    def _fill(self, c):
        for i in range(6):
            c.write_telemetry(_payload(inlet=22.0 if i < 4 else 30.0, pue=1.04 + 0.01 * i))
        c.write_telemetry(_payload(facility="DC-EU-01"))

    def test_history_is_served_from_memory_after_the_cloud_fails(self, cloud_client):
        self._fill(cloud_client)
        rows = cloud_client.get_telemetry_history("DC-EAST-01", limit=100)
        assert len(rows) == 6 and all(r["facility_id"] == "DC-EAST-01" for r in rows)

    def test_a_read_that_itself_hits_the_outage_falls_back_instead_of_returning_empty(self):
        c = TimestreamClient(local_mode=False)
        c._write_client, c._query_client = type("Ok", (), {"write_records": lambda self, **k: None})(), _DownQuery()
        for i in range(4):
            c.write_telemetry(_payload(pue=1.05))
        # cloud writes work, but the query fails: the very first read must still return data
        assert len(c.get_telemetry_history("DC-EAST-01", limit=100)) == 4
        assert c._cloud_failures == 1

    def test_analytics_fall_back_too(self, cloud_client):
        self._fill(cloud_client)
        assert cloud_client.get_latest_pue("DC-EAST-01") == pytest.approx(1.065, abs=0.02)
        assert cloud_client.get_sla_violation_rate("DC-EAST-01") == pytest.approx(2 / 6)
        snap = cloud_client.get_spatial_snapshot()
        assert {r["facility_id"] for r in snap} == {"DC-EAST-01", "DC-EU-01"}

    def test_local_mode_never_touches_the_cloud(self):
        c = TimestreamClient(local_mode=True)
        c.write_telemetry(_payload())
        assert c._cloud_ready() is False and len(c.get_telemetry_history("DC-EAST-01")) == 1
