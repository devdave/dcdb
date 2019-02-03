import pytest
import time
import pathlib
import logging
import dataclasses as dcs
from unittest.mock import MagicMock

from dcdb import dcdb
from dcdb import migrate_lib as ml

LOG = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def conn(request):
    ts = int(time.time())
    db_name = f"{request.function.__name__}.sqlite3"
    filepath_ts = pathlib.Path(__file__).parent / "db" / f"{ts}_{db_name}"
    filepath = pathlib.Path(__file__).parent / "db" / db_name

    if filepath.exists():
        try:
            filepath.unlink()
        except PermissionError:
            filepath = filepath_ts

        for olddb in (pathlib.Path(__file__).parent / "db").glob(f"*_{db_name}"):
            try:
                olddb.unlink()
            except PermissionError:
                pass



    LOG.debug(f"Test DB @ {filepath}")
    connection = dcdb.DBConnection(filepath)
    yield connection
    connection.close()


def test_Schema__correct_manifest_empty_db(conn):
    schema = ml.Schema(conn)

    assert schema.tables == []


def test_Schema_correct_manifest_with_tables(conn):

    @dcs.dataclass()
    class Foo:
        pass

    @dcs.dataclass()
    class Bar:
        pass

    models = MagicMock()
    models.Foo = Foo
    models.Bar = Bar

    conn.bind(Foo)
    conn.bind(Bar)
    schema = ml.Schema(conn)

    assert schema.tables == ["Foo", "Bar"]
    assert schema.compare(models).no_missing == True


def test_Schema_compare__detects_missing_model_def(conn):

    @dcs.dataclass()
    class Foo:
        pass

    @dcs.dataclass()
    class Bar:
        pass

    models = MagicMock()
    models.Foo = Foo


    conn.bind(Foo)
    conn.bind(Bar)
    schema = ml.Schema(conn)

    actual = schema.compare(models) # type: ml.DiffSets

    assert actual.no_missing == False
    assert "Bar" in actual.missing_definitions
    assert actual.missing_database == set()