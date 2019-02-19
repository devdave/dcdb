# pylint: disable=C0111,C0112,C0301,C0103,W0621,C0102
# TODO remake .pylintrc?  I only want some of these disables for tests though
"""
    Tests for DCDB
"""

import logging
import pytest

import dcdb

from dataclasses import dataclass, fields, field
import enum
import pathlib

import datetime as dt

import typing

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


@pytest.fixture()
def conn2(connection: dcdb.DBConnection):
    """

    :param connection: dcdb.DBConnection uses the connection fixture
    :return:
    """

    @dataclass()
    class Widget:
        name: str
        age: int
        panacea: bool

    connection.bind(Widget)

    return connection


def test_dcdb_cast_to_database_type__fix_ordering_error():
    """
    Odd test but written to isolate a bug

    """

    value = "Bob"
    result = dcdb.cast_to_database(value, str)
    assert isinstance(result, str)
    assert value == result





def test_FieldPickled__AND__dcdb_cast_to_database___works():
    import pickle

    @dataclass
    class Foo:
        colors: dcdb.FieldPickled

    test_value = {"red": 1, "blue": 2, "complicated": [1, 2, "this isn't a color"]}
    test_subject = Foo(test_value)

    foo_fields = fields(Foo)
    colors_field = fields(Foo)[0]
    result = dcdb.cast_to_database(test_value, colors_field.type)
    assert isinstance(result, bytes)
    assert result == pickle.dumps(test_value)


def test_dcdb__cast_to_database():
    class Switch(enum.Enum):
        OFF = 0
        ON = 1

        @classmethod
        def To(cls, value, _):
            return value.name

        @classmethod
        def From(cls, value, _):
            return cls.__members__[value]

    test = Switch.OFF
    expected = "OFF"
    actual = dcdb.cast_to_database(test, Switch)
    assert expected == actual


def test_TransformDatetimeType__works(connection):

    import datetime as dt

    @dataclass()
    class Foo:
        daytime: dcdb.TransformDatetimeType(dt.datetime, "%Y-%m-%d T%H:%M:%S") = None # type: dt.datetime
        day: dcdb.TransformDatetimeType(dt.date, "%Y-%m-%d") = None # type: dt.date
        handm: dcdb.TransformDatetimeType(dt.time, "%H:%M:%S") = None # type: dt.time

    connection.bind(Foo)

    foo_tbl = connection.t.Foo # type: Foo

    x = foo_tbl() # type: Foo
    x_id = x.id

    assert x.daytime is None
    assert x.day is None
    assert x.handm is None

    x.daytime = dt.datetime(1999,1,1,1,10,30)
    x.save()


    y = foo_tbl.Get("id=?", x_id) # type: Foo
    assert y.daytime.hour == 1
    assert y.daytime.year == 1999
    assert y.daytime.second == 30

    y.handm = dt.time(11,55,33)
    y.save()

    z = foo_tbl.Get("id=?", x_id) # type: Foo
    assert z.handm.hour == 11
    assert z.handm.minute == 55
    assert z.handm.second == 33







def test_FieldPickled_AND_FieldJSON__works_as_expected(connection):
    @dataclass
    class Foo:
        bar: dcdb.dcdb.FieldPickled
        name: str

    @dataclass()
    class Bar:
        name: str
        blank: dcdb.FieldJSON = None

    connection.bind(Foo)
    connection.bind(Bar)

    test_value = {"a": 1, 2: "b", "three": "c"}
    expected_json = {"a": 1, '2': "b", "three": "c"}

    record = connection.t.Foo(bar=test_value, name="Test thing")

    assert record.bar == test_value, repr(record.bar)

    record2 = connection.t.Bar(name="Bob")

    assert record2.blank == None

    record3 = connection.t.Bar(name="Bob", blank=test_value)

    assert record3.blank == expected_json


def test_FieldJSON__works_when_assigned_raw_string(connection):

    import json

    @dataclass()
    class Thing:
        widget: dcdb.FieldJSON = None

    connection.bind(Thing)

    actual = dict(foo="bar", nums=123, complex={"a":2, "123":"abc"})
    raw_actual = json.dumps(actual)

    record = connection.t.Thing(widget=raw_actual)
    assert record.widget == actual


def test_FieldJSON__works_when_is_set_to_None(connection):
    @dataclass()
    class Thing:
        widget: dcdb.FieldJSON = None

    connection.bind(Thing)

    record = connection.t.Thing(widget=None)
    assert record.widget == None





def test_Transformers__handles_decimal():
    import decimal

    dcdb.Transformers.Set(decimal.Decimal, lambda v, t: str(v), lambda v, t: t(v))

    actual = 1.0
    to_val = "1.0"

    assert dcdb.Transformers.Has(decimal.Decimal) is True
    assert dcdb.Transformers.From(to_val, decimal.Decimal) == actual
    assert dcdb.Transformers.To(actual, decimal.Decimal) == to_val


def test_dcdb__cast_to_database_AND_cast_from_database___column_as_enum(connection):
    import enum

    class EnumFlags(enum.Enum):
        OFF = 0
        ON = 1

        @classmethod
        def To(cls, value, _):
            return value.name

        @classmethod
        def From(cls, value, _):
            return cls.__members__[value]

    @dataclass()
    class Test:
        enum_column: EnumFlags = EnumFlags.OFF

    connection.bind(Test)
    default_record = connection.t.Test()
    off_record = connection.t.Test(enum_column=EnumFlags.OFF)
    on_record = connection.t.Test(enum_column=EnumFlags.ON)

    assert off_record.enum_column == EnumFlags.OFF
    assert on_record.enum_column == EnumFlags.ON