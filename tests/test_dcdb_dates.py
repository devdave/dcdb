
import dcdb
from dataclasses import dataclass
import pathlib
import datetime as dt
import pytest
import logging



LOG = logging.getLogger(__name__)


@pytest.fixture
def connection(request) -> dcdb.DBConnection:
    """
        Fixture used to speed up testing
    """
    db_file = pathlib.Path(__file__).parent / "db" / f"{request.function.__name__}.sqlite"
    if db_file.exists():
        db_file.unlink()

    return dcdb.DBConnection(str(db_file))


def test_Transformers_AND_Datetime_dot_date():
    actual_date = dt.date(2018, 3, 5)
    actual_date_str = "2018-03-05"
    assert dcdb.Transformers.To(actual_date, dt.date) == actual_date_str
    assert dcdb.Transformers.To(actual_date_str, dt.date) == actual_date_str

    assert dcdb.Transformers.From(actual_date_str, dt.date) == actual_date



def test_DefaultLocalTime___allow_default_time(connection):

    import datetime

    @dataclass()
    class ThatTime:
        created_on: dcdb.DefaultLocalTime() = dcdb.DefaultLocalTime.ColDef

    connection.bind(ThatTime)


    now = datetime.datetime.now()
    created_now = connection.t.ThatTime()

    assert created_now.created_on.year == now.year
    assert created_now.created_on.month == now.month
    assert created_now.created_on.day == now.day
    assert created_now.created_on.hour == now.hour
    assert now.minute+1 > created_now.created_on.minute > now.minute - 1



def test_dcdb__cast_to_database_AND_cast_from_database__handle_datetime(connection):
    import datetime

    @dataclass()
    class TimePiece:
        a_date: datetime.date
        a_time: datetime.time
        a_datetime: datetime.datetime

    connection.bind(TimePiece)

    test_datetime = datetime.datetime(1955, 11, 5, 11, 10, 54, 67345)
    test_date = datetime.date(1941, 12, 7)
    test_time = datetime.time(10, 20, 11)

    record = connection.t.TimePiece(a_date=test_date, a_datetime=test_datetime, a_time=test_time)

    assert record.a_date == test_date
    assert record.a_time == test_time
    assert record.a_datetime == test_datetime