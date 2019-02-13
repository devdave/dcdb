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


@dataclass
class Widget:
    # __table__ = "widgets"
    name: str
    age: int
    panacea: bool = False  # in retrospect this makes no sense but I am just going to keep using this word





def test_DBConnection___connects():
    my_conn = dcdb.DBConnection(":memory:")
    result = my_conn.handle().execute("SELECT 1=1").fetchone()
    assert result[0] == 1





def test_DBTableRegistry___mk_bound(connection):
    @dataclass()
    class Widget:
        name: str
        age: int
        panacea: bool

    connection.bind(Widget)
    res = connection.handle().execute(
        "SELECT name FROM sqlite_master WHERE type='table' and name = 'Widget' ORDER BY name;")
    name = res.fetchone()
    assert name[0] == "Widget"
    assert connection.registry.Widget is not None
    rows = connection.execute("PRAGMA table_info(Widget)").fetchall()

    assert rows[0]['name'] == "id"
    assert rows[0]['type'] == "INTEGER"

    assert rows[1]['name'] == "name"
    assert rows[1]['type'] == "TEXT"

    assert rows[2]['name'] == "age"
    assert rows[2]['type'] == "INTEGER"

    # Sqlite coerces bools to intege
    # which is kind of a stretch as basically sqlite coerces everything into a string

    assert rows[3]['name'] == "panacea"
    assert rows[2]['type'] == "INTEGER"


def test_DBCommonTable___ishashable(conn2):
    x = conn2.t.Widget(name="Bob", age=10, panacea=True)
    y = conn2.t.Widget(name="Alive", age=12, panacea=False)
    x2 = conn2.t.Widget.Get("name=?", "Bob")
    assert (y == x) is False
    assert (x == x) is True
    assert (x == x2) is True
    records = set()
    assert x.__hash__ is not None
    records.add(x)
    records.add(y)


def test_DBConnection__bind_blowsup_with_typeerror_on_nondataclass(connection):
    class Bogus:
        foo: str = "bar"

    with pytest.raises(TypeError):
        connection.bind(Bogus)


def test_DBConnection___binds_multiple_tables(connection):
    @dataclass()
    class Foo:
        name: str

    @dataclass()
    class Bar:
        name: str
        num: int

    @dataclass()
    class Bob:
        last_name: str
        age: int

    (MyFoo, MyBar, MyBob,) = connection.binds(Foo, Bar, Bob)

    for (test_cls, db_cls,) in ((Foo, MyFoo,), (Bar, MyBar,), (Bob, MyBob,)):
        assert issubclass(db_cls, test_cls)
        assert issubclass(db_cls, dcdb.DBCommonTable)


def test_DBCommonTable__assure_post_init_overload_works(connection):
    @dataclass()
    class Foo:
        def __post_init__(self, *args, **kwargs):
            super().__post_init__(*args, **kwargs)
            LOG.debug("I was called")

    connection.bind(Foo)

    record = connection.tables.Foo()


def test_make_complex_widget(connection):
    from typing import ClassVar
    @dataclass
    class ComplexWidget:
        # by definition of how dataclasses work, these cannot be empty/unassigned/null so don't bother telling sqlite to enforce that.
        #
        name: str
        age: int

        NameAgeUnique: dcdb.TableDef = "CONSTRAINT NameAgeUnique UNIQUE(name,age)"

    connection.bind(ComplexWidget)
    res = connection.handle().execute("""PRAGMA index_info(sqlite_autoindex_ComplexWidget_1)""")

    index_rows = res.fetchall()
    """
        [0, 1, 'name']
        [1, 2, 'age']
    """
    assert len(index_rows) == 2
    assert index_rows[0][2] == "name"
    assert index_rows[1][2] == "age"


def test_db_dataclass_throws_error_on_missing_param(conn2):
    with pytest.raises(ValueError):
        conn2.t.Widget(id=1)

    with pytest.raises(ValueError):
        conn2.t.Widget(id=1, name="Bob")

    with pytest.raises(ValueError):
        conn2.t.Widget(id=1, name="Bob", age=55)

    with pytest.raises(dcdb.IntegrityError):
        conn2.t.Widget()

    with pytest.raises(dcdb.IntegrityError):
        conn2.t.Widget(name="Bob")

    with pytest.raises(dcdb.IntegrityError):
        conn2.t.Widget(name="Bob", age=55)

    record = conn2.t.Widget(name="Bob", age=55, panacea=True)
    assert record.id == 1  # asserts this is the first record added to the table, therefore assumes only record
    assert record.name == "Bob"
    assert record.age == 55
    assert record.panacea == True


def test_DBTableRegistry__create(conn2):
    with pytest.raises(dcdb.IntegrityError):
        conn2.t.Widget(name="Bob", age=55)

    conn2.t.Widget(name="Bob", age=55, panacea=False)
    row = conn2.execute("SELECT * FROM Widget LIMIT 1").fetchone()

    assert row['id'] == 1
    assert row['name'] == "Bob"
    assert row['age'] == 55


def test_DBTableProxy_InsertMany(conn2):
    MyWidget = conn2.t.Widget
    MyWidget.Insert_many(
        dict(name="Bob", age=10, panacea=True),
        dict(name="Joe", age=22, panacea=False),
        dict(name="Steve", age=35, panacea=True)
    )

    record = conn2.execute("SELECT COUNT(*) FROM Widget").fetchone()
    assert record[0] == 3

    bob = MyWidget.Get("name=?", "Bob")
    Joe = MyWidget.Get("name=?", "Joe")
    Steve = MyWidget.Get("name=?", "Steve")
    assert bob.age == 10
    assert Joe.name == "Joe"  # this seems redundent


def test_DBCommonTable_AND_DBRegistry__create_with_default_factory(connection):
    @dataclass()
    class Test:
        foo: str = "Hello"
        bar: int = dcdb.ColumnDef(python=None, database="TIMESTAMP DEFAULT (strftime('%s','now'))")

    connection.bind(Test)

    instance = connection.t.Test.Create()


def test_DBTableRegistry__create(conn2):
    with pytest.raises(ValueError):
        record = conn2.t.Widget(id=1)

    with pytest.raises(dcdb.IntegrityError):
        record = conn2.t.Widget(name="Bob", age=55)

    record = conn2.t.Widget(name="Bob", age=55, panacea=True)
    record.save()
    assert record.id == 1
    assert record.name == "Bob"
    assert record.age == 55
    assert record.panacea == True

    record = conn2.t.Widget(name="Bob", age=55, panacea=False)
    record.save()
    assert record.id == 2
    assert record.panacea == False


def test_DBConnection__data_integrity_from_sqlite(conn2):
    with pytest.raises(dcdb.IntegrityError):
        conn2.execute("INSERT INTO Widget(name,age) VALUES ('Bob', 33)")


def test_DBCommonTable__Select(conn2):
    conn2.execute("INSERT INTO Widget(name,age,panacea) VALUES ('Bob', 33, 1)")
    row = conn2.t.Widget.Select("name=?", "Bob").fetchone()

    assert row.name == "Bob"
    assert row.age == 33
    assert row.panacea == 1


def test_DBRegisteredTable_AND_DBCommonTable___Insert_Select_Get(conn2):
    MyWidget = conn2.t.Widget
    MyWidgetClass = MyWidget.bound_cls

    MyWidget(name="Bob", age=44, panacea=False)

    record = MyWidget.Select("name=?", "Bob").fetchone()

    assert record.name == "Bob"
    assert record.age == 44
    assert record.panacea == False

    record2 = MyWidget.Get("name=?", "Bob")
    assert isinstance(record2, MyWidgetClass)
    assert record == record2


def test_DBCommonTable_update(conn2):
    conn2.bind(Widget)

    # TODO should I consider it a problem that setting id skips saving the actual record?
    #  record = conn2.t.Widget(id=10, name="Joe", age=22, panacea=True)
    record = conn2.t.Widget(name="Joe", age=22, panacea=True)
    assert record.name == "Joe"
    assert record.age == 22
    assert record.panacea is True

    record.panacea = False
    record.age = 33
    record.update()
    same_record = conn2.t.Widget.Get("id=?", record.id)
    assert isinstance(same_record, conn2.t.Widget.bound_cls)
    assert same_record.panacea is False
    assert same_record.age == 33
    assert same_record.id == record.id


def test_DBCommonTable_Update__bulk_update(conn2):
    # Populate the db
    conn2.t.Widget(name="Bob", age=13, panacea=True)
    conn2.t.Widget(name="Joe", age=22, panacea=True)
    conn2.t.Widget(name="Sue", age=34, panacea=True)
    conn2.t.Widget(name="Mary", age=44, panacea=True)
    conn2.t.Widget(name="It", age=9999, panacea=True)

    # Bulk update roughly half
    cursor = conn2.t.Widget.Update(["age<?", 35], panacea=False)

    # Sanity check they are correct
    assert conn2.t.Widget.Get("name=?", "Bob").panacea is False
    assert conn2.t.Widget.Get("name=?", "Joe").panacea is False
    assert conn2.t.Widget.Get("name=?", "Sue").panacea is False
    assert conn2.t.Widget.Get("name=?", "Mary").panacea is True
    assert conn2.t.Widget.Get("name=?", "It").panacea is True





def test_DBCommonTable_hashing_equals(conn2):
    record1 = conn2.t.Widget(name="Alice", age=35, panacea=False)
    record2 = conn2.t.Widget.Get("name=?", "Alice")
    record3 = conn2.t.Widget.Get("age=?", 35)
    assert record1.id == record2.id
    assert record1 == record2
    assert record2 == record3
    assert record3 == record1

    @dataclass()
    class Foo:
        height: int
        width: int

    conn2.bind(Foo)
    record4 = conn2.t.Foo(height=12, width=22)
    record5 = conn2.t.Foo.Get("height=?", 12)
    record6 = conn2.t.Foo.Get("width=?", 22)
    assert record4 != record1
    assert record5 != record2
    assert record6 != record3
    assert record6 == record4
    assert record4 == record5
    assert record5 != record3


def test_DBTableRegistry___works(connection):
    MyWidget = connection.bind(Widget)
    SameWidget = connection.t.Widget
    assert issubclass(MyWidget, Widget)
    assert issubclass(SameWidget.bound_cls, Widget)


def test_DBTablesRegistry___exception_on_missing_table(connection):
    with pytest.raises(RuntimeError):
        connection.t.NotATable




def test_DBCommonTable___fix_error_columns_mismatch(conn2):
    """
        At one point, the table/record creation mechanism
        used the internal fields dictionary for record creation
        which had the unintended consequence of creating records
        with fields that were not in order and therefore value mismatched.

    """

    @dataclass
    class OtherWidget:
        length: int
        height: int
        weight: int
        name: str

    conn2.bind(OtherWidget)
    thingamajig = conn2.t.OtherWidget(name="Thinga 2000", length=10, height=1, weight=8)
    assert thingamajig.name == "Thinga 2000"
    assert thingamajig.length == 10
    assert thingamajig.height == 1
    assert thingamajig.weight == 8


def test_mutation_tracking(conn2: dcdb.DBConnection):
    @dataclass
    class OtherWidget:
        length: int
        height: int
        weight: int
        name: str

    conn2.bind(OtherWidget)
    bob = conn2.t.Widget(name="Bob", age=44, panacea=False)
    thingamajig = conn2.t.OtherWidget(name="Thinga 2000", length=10, height=1, weight=8)

    assert bob._is_dirty is False
    assert thingamajig._is_dirty is False

    # Known and intended gotcha
    thingamajig.height = 5
    assert thingamajig._is_dirty is False

    with conn2.track_changes():
        bob.age = 22
        assert bob._is_dirty is True
        assert len(conn2._dirty_records) == 1

        thingamajig.weight = 12
        thingamajig.name = "New Hotness 2.0"
        assert len(conn2._dirty_records) == 2
        assert thingamajig._is_dirty is True

    assert len(conn2._dirty_records) == 0
    assert bob._is_dirty is False
    assert thingamajig._is_dirty is False

    bob_copy = conn2.t.Widget.Get("id=?", bob.id)
    thing_copy = conn2.t.OtherWidget.Get("id=?", thingamajig.id)

    assert bob_copy.age == 22
    assert thingamajig.weight == 12
    assert thingamajig.name == "New Hotness 2.0"
    # Intentional and intended, change tracking is only available when explicitly tracking
    assert thingamajig.height == 5











def main():
    import sys
    import pathlib as pl

    my_path = pl.Path(__file__).resolve().parent
    sys.path.append(str(my_path))

    pytest.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
